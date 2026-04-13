"""
Visual Pipeline - Engine  (ViT Edition)
========================================
Training/validation/inference loops with:
  - torch.amp mixed precision (bfloat16 on Ampere+, float16 on older)
  - Exponential Moving Average (EMA) of model weights
  - Soft-label cross-entropy for MixUp support
  - mAP via sklearn
  - Gradient norm clipping + anomaly detection
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from sklearn.metrics import average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ──────────────────────────────────────────────────────────────
#  AverageMeter
# ──────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)

    def __str__(self):
        return f"{self.name}={self.avg:.4f}"


# ──────────────────────────────────────────────────────────────
#  EMA
# ──────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    Use `ema.model` for validation — consistently 0.5-1% better mAP.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(
            self.model.parameters(), model.parameters()
        ):
            ema_p.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.model.state_dict()


# ──────────────────────────────────────────────────────────────
#  Soft-label cross entropy  (for MixUp)
# ──────────────────────────────────────────────────────────────

class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard cross entropy with class weights and optional label smoothing.
    Accepts hard integer labels only (no MixUp soft labels).
    """
 
    def __init__(
        self,
        weight          : Optional[Tensor] = None,
        label_smoothing : float = 0.0,
    ):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            weight          = weight,
            label_smoothing = label_smoothing,
        )
 
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.loss_fn(logits, targets.long())
 

# ──────────────────────────────────────────────────────────────
#  AMP dtype detection
# ──────────────────────────────────────────────────────────────

def get_amp_dtype(device: torch.device) -> torch.dtype:
    """Use bfloat16 on Ampere+ (sm_80+), float16 otherwise."""
    if device.type != "cuda":
        return torch.float32
    major = torch.cuda.get_device_capability(device)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


# ──────────────────────────────────────────────────────────────
#  mAP
# ──────────────────────────────────────────────────────────────

def compute_map(labels: list[int], scores: list[float]) -> float:
    if not HAS_SKLEARN or len(set(labels)) < 2:
        preds = [int(s > 0.5) for s in scores]
        return sum(p == l for p, l in zip(preds, labels)) / max(len(labels), 1)
    return float(average_precision_score(labels, scores))


# ──────────────────────────────────────────────────────────────
#  Train
# ──────────────────────────────────────────────────────────────

def train(
    loader      : DataLoader,
    model       : nn.Module,
    criterion   : nn.Module,
    optimizer   : torch.optim.Optimizer,
    epoch       : int,
    scaler      : Optional[GradScaler]  = None,
    ema         : Optional[ModelEMA]    = None,
    device      : torch.device          = torch.device("cuda"),
    amp_dtype   : torch.dtype           = torch.float16,
    mixup_fn    : Optional[callable]    = None,
    log_interval: int = 50,
    grad_clip   : float = 1.0,
    scheduler=None,
) -> float:

    model.train()
    losses = AverageMeter("loss")
    t0 = time.time()

    pbar = tqdm(
        loader,
        desc        = f"Epoch {epoch:03d}",
        leave       = False,
        dynamic_ncols= True,
    )

    for step, (clips, labels) in enumerate(pbar):
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # MixUp (optional — only during training)
        if mixup_fn is not None:
            clips, labels = mixup_fn(clips, labels)

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device.type == "cuda"

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(clips)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        else:
            logits = model(clips)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"  [Skip] Bad logits at step {step} — skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [Skip] NaN loss at step {step}  labels={labels}")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        if ema is not None:
            ema.update(model)

        losses.update(loss.item(), clips.size(0))
        pbar.set_postfix(loss=f"{losses.avg:.4f}")

        if (step + 1) % log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  ep={epoch} step={step+1}/{len(loader)}  "
                  f"loss={losses.avg:.4f}  lr={lr:.2e}  "
                  f"t={time.time()-t0:.0f}s")

    return losses.avg


# ──────────────────────────────────────────────────────────────
#  Validate
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    loader   : DataLoader,
    model    : nn.Module,
    device   : torch.device = torch.device("cuda"),
    amp_dtype: torch.dtype  = torch.bfloat16,
    mode     : str          = "val",
) -> float:

    model.eval()
    all_labels : list[int]   = []
    all_scores : list[float] = []
    total_loss = 0.0
    total_n    = 0
    criterion  = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc=f"  {mode}", leave=False, dynamic_ncols=True)

    for clips, labels in pbar:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ensure labels are always 1D
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        with torch.autocast(
            device_type = device.type,
            dtype       = amp_dtype if device.type == "cuda" else torch.float32,
            enabled     = device.type == "cuda",
        ):
            logits = model(clips)

        if torch.isnan(logits).any():
            continue

        loss = criterion(logits, labels.long())
        total_loss += loss.item() * clips.size(0)
        total_n    += clips.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        all_scores.extend(probs)
        all_labels.extend(labels.cpu().tolist())   # ← always collected together

        pbar.set_postfix(loss=f"{total_loss/max(total_n,1):.4f}")

    avg_loss = total_loss / max(total_n, 1)
    mAP      = compute_map(all_labels, all_scores)
    acc      = sum(
        int(s > 0.5) == int(l)
        for s, l in zip(all_scores, all_labels)
    ) / max(len(all_labels), 1)
    pos_preds = sum(1 for s in all_scores if s > 0.5)
    if all_scores:
        score_min  = min(all_scores)
        score_max  = max(all_scores)
        score_mean = sum(all_scores) / len(all_scores)
    else:
        score_min = score_max = score_mean = 0.0

    print(f"  [{mode}] loss={avg_loss:.4f}  mAP={mAP:.4f}  "
          f"acc={acc:.4f}  pred_pos={pos_preds}/{len(all_scores)}  "
          f"score[min/mean/max]={score_min:.4f}/{score_mean:.4f}/{score_max:.4f}")
    return mAP

# ──────────────────────────────────────────────────────────────
#  Infer
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(
    loader     : DataLoader,
    model      : nn.Module,
    output_path: str,
    device     : torch.device = torch.device("cuda"),
    amp_dtype  : torch.dtype  = torch.float16,
) -> dict:

    model.eval()
    results: dict[str, list] = {}

    pbar = tqdm(loader, desc="  Infer", leave=False, dynamic_ncols=True)

    for clips, infos in pbar:
        clips = clips.to(device, non_blocking=True)

        with torch.autocast(
            device_type = device.type,
            dtype       = amp_dtype if device.type == "cuda" else torch.float32,
            enabled     = device.type == "cuda",
        ):
            logits = model(clips)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

        for prob, info in zip(probs, infos):
            uid = info["uid"]
            results.setdefault(uid, []).append({
                "score"    : round(prob, 4),
                "fid2pred" : info.get("fid2pred", []),
            })

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "predictions.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  [Infer] {len(results)} UIDs → {out_file}")
    return results


# ──────────────────────────────────────────────────────────────
#  Checkpoint helpers
# ──────────────────────────────────────────────────────────────

def save_checkpoint(
    state    : dict,
    save_path: str,
    is_best  : bool = False,
    filename : str  = "checkpoint.pth",
):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, filename)
    torch.save(state, path)
    if is_best:
        best = os.path.join(save_path, "best_model.pth")
        torch.save(state, best)
        print(f"  ✓ Best model → {best}")


def load_checkpoint(
    model      : nn.Module,
    ckpt_path  : str,
    optimizer  : Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device     : torch.device = torch.device("cuda"),
) -> tuple[nn.Module, int, float]:

    print(f"[Checkpoint] Loading {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    epoch = ckpt.get("epoch", 0)
    mAP   = ckpt.get("mAP",   0.0)

    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as e:
            print(f"  ⚠️  Could not load optimizer state: {e}")
            print(f"     Starting with fresh optimizer (common when model/config changed)")
    if scheduler and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
            print(f"  Scheduler state restored")
        except (ValueError, AttributeError) as e:
            print(f"  ⚠️  Could not load scheduler state: {e}")
    print(f"  Resumed epoch={epoch}  mAP={mAP:.4f}")
    return model, epoch, mAP
