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
import torch.nn.functional as F
try:
    from sklearn.metrics import average_precision_score, roc_auc_score,precision_recall_curve, f1_score
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

class WeightedBinaryCrossEntropyLoss(nn.Module):
    """
    Weighted BCEWithLogits for binary TTM.
    Supports hard labels [B] and soft one-hot labels [B, 2] from MixUp.
    """

    def __init__(
        self,
        weight          : Optional[Tensor] = None,
        label_smoothing : float = 0.0,
    ):
        super().__init__()
        self.label_smoothing = float(max(0.0, min(1.0, label_smoothing)))

        neg_w = 1.0
        pos_w = 1.0
        if weight is not None and len(weight) >= 2:
            neg_w = float(weight[0].item())
            pos_w = float(weight[1].item())
        self.register_buffer("neg_weight", torch.tensor(neg_w, dtype=torch.float32))
        self.register_buffer("pos_weight", torch.tensor(pos_w, dtype=torch.float32))

    @staticmethod
    def _binary_logit(logits: Tensor) -> Tensor:
        if logits.dim() == 2 and logits.size(1) == 2:
            return logits[:, 1] - logits[:, 0]
        if logits.dim() == 2 and logits.size(1) == 1:
            return logits[:, 0]
        if logits.dim() == 1:
            return logits
        raise ValueError(f"Unsupported logits shape for BCE: {tuple(logits.shape)}")

    @staticmethod
    def _positive_target(targets: Tensor) -> Tensor:
        if targets.dim() == 2 and targets.size(1) >= 2:
            return targets[:, 1].float()
        return targets.float().view(-1)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        pos_logit = self._binary_logit(logits)
        y = self._positive_target(targets)

        if self.label_smoothing > 0:
            y = y * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss = nn.functional.binary_cross_entropy_with_logits(
            pos_logit,
            y,
            reduction="none",
        )

        neg_w = self.neg_weight.to(loss.device)
        pos_w = self.pos_weight.to(loss.device)
        sample_w = y * pos_w + (1.0 - y) * neg_w
        return (loss * sample_w).mean()


# Backward-compatible alias used by existing run.py imports.
WeightedCrossEntropyLoss = WeightedBinaryCrossEntropyLoss
class BinaryFocalLoss(nn.Module):
    """
    Focal Loss adapted for models outputting [batch_size, 2] logits.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits shape: [B, 2] (raw scores for class 0 and class 1)
        # targets shape: [B] (integer labels 0 or 1)
        
        # 1. Calculate standard Cross Entropy Loss (this handles the [8, 2] vs [8] mismatch)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 2. Get the probability of the true class (pt)
        pt = torch.exp(-ce_loss)
        
        # 3. Apply the alpha weighting dynamically
        # If target is 1 (Positive), use alpha. If target is 0 (Negative), use 1 - alpha.
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        
        # 4. Calculate the final Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
def find_optimal_threshold(
    labels: list[int],
    scores: list[float],
    target_precision: float = 0.55,
) -> tuple[float, float]:
    """Find the threshold with maximum recall subject to a precision floor.

    Returns:
        (best_threshold, achieved_recall)
    """
    if not HAS_SKLEARN or len(labels) == 0 or len(scores) == 0:
        return 0.5, 0.0

    # PR-curve requires both classes to be present.
    if len(set(labels)) < 2:
        return 0.5, 0.0

    try:
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    except ValueError:
        return 0.5, 0.0

    # precision_recall_curve returns one extra terminal point with no matching
    # threshold, so only search over threshold-aligned PR points here.
    aligned_precisions = precisions[:-1]
    aligned_recalls = recalls[:-1]

    valid_indices = [
        idx for idx, precision in enumerate(aligned_precisions)
        if precision >= target_precision
    ]

    if not valid_indices:
        # Fallback: if the model cannot satisfy the precision floor, use the
        # threshold-aligned point with the highest precision.
        best_idx = int(aligned_precisions.argmax())
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
        return float(best_thresh), float(aligned_recalls[best_idx])

    best_valid_idx = max(valid_indices, key=lambda idx: aligned_recalls[idx])
    best_thresh = thresholds[best_valid_idx] if best_valid_idx < len(thresholds) else 1.0

    return float(best_thresh), float(aligned_recalls[best_valid_idx])
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
    grad_accum_steps: int = 1,
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

    grad_accum_steps = max(int(grad_accum_steps), 1)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        if len(batch) == 3:
            clips, track, labels = batch
            track = track.to(device, non_blocking=True)
        else:
            clips, labels = batch
            track = None

        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # MixUp (optional — only during training)
        if mixup_fn is not None:
            clips, labels = mixup_fn(clips, labels)

        use_amp = scaler is not None and device.type == "cuda"
        is_update_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader))
        loss_scale = 1.0 / grad_accum_steps

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                try:
                    logits = model(clips, track)
                except TypeError:
                    logits = model(clips)
                loss   = criterion(logits, labels) * loss_scale

            if not torch.isfinite(logits).all():
                print(f"  [Skip] Bad logits at step {step} (AMP) — skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            if not torch.isfinite(loss):
                print(f"  [Skip] NaN/Inf loss at step {step} (AMP) — skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
        else:
            try:
                logits = model(clips, track)
            except TypeError:
                logits = model(clips)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"  [Skip] Bad logits at step {step} — skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss = criterion(logits, labels) * loss_scale

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [Skip] NaN loss at step {step}  labels={labels}")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

        if is_update_step:
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            if ema is not None:
                ema.update(model)

        losses.update(loss.item() * grad_accum_steps, clips.size(0))
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
    criterion: Optional[nn.Module] = None,
) -> float:

    model.eval()
    all_labels : list[int]   = []
    all_scores : list[float] = []
    total_loss = 0.0
    total_n    = 0
    criterion = criterion or WeightedBinaryCrossEntropyLoss()

    pbar = tqdm(loader, desc=f"  {mode}", leave=False, dynamic_ncols=True)

    for batch in pbar:
        if len(batch) == 3:
            clips, track, labels = batch
            track = track.to(device, non_blocking=True)
        else:
            clips, labels = batch
            track = None

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
            try:
                logits = model(clips, track)
            except TypeError:
                logits = model(clips)

        if torch.isnan(logits).any():
            continue

        loss = criterion(logits, labels)
        total_loss += loss.item() * clips.size(0)
        total_n    += clips.size(0)

        if logits.dim() == 2 and logits.size(1) == 2:
            pos_logit = logits[:, 1] - logits[:, 0]
        elif logits.dim() == 2 and logits.size(1) == 1:
            pos_logit = logits[:, 0]
        else:
            pos_logit = logits.view(-1)
        probs = torch.sigmoid(pos_logit).cpu().tolist()
        all_scores.extend(probs)
        all_labels.extend(labels.cpu().tolist())   # ← always collected together

        pbar.set_postfix(loss=f"{total_loss/max(total_n,1):.4f}")

    avg_loss = total_loss / max(total_n, 1)
    mAP      = compute_map(all_labels, all_scores)
    acc      = sum(
        int(s > 0.5) == int(l)
        for s, l in zip(all_scores, all_labels)
    ) / max(len(all_labels), 1)
    opt_thresh, opt_recall = find_optimal_threshold(all_labels, all_scores)
    pos_preds_05  = sum(1 for s in all_scores if s > 0.5)
    pos_preds_opt = sum(1 for s in all_scores if s > opt_thresh)

    # compute AUC first
    auc = 0.5
    if all_scores:
        score_min  = min(all_scores)
        score_max  = max(all_scores)
        score_mean = sum(all_scores) / len(all_scores)
        try:
            auc = float(roc_auc_score(all_labels, all_scores))
        except ValueError:
            auc = 0.5
    else:
        score_min = score_max = score_mean = 0.0

    # now print with auc already defined
    print(f"  [{mode}] loss={avg_loss:.4f}  mAP={mAP:.4f}  AUC={auc:.4f}  acc={acc:.4f}")
    print(f"  threshold=0.50:           pred_pos={pos_preds_05}/{len(all_scores)}")
    print(f"  threshold={opt_thresh:.3f} (optimal): pred_pos={pos_preds_opt}/{len(all_scores)}  recall={opt_recall:.4f}")
    print(f"  score[min/mean/max]={score_min:.4f}/{score_mean:.4f}/{score_max:.4f}")

    return mAP, auc

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
