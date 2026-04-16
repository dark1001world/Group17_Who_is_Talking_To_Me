"""
train_audio.py  –  WhisperTTM training loop
Fixes vs previous version:
  - fp32 dtype handled correctly (no float16 fallback)
  - labels stay as LongTensor (not one-hot) for CrossEntropyLoss + metrics
  - criterion built once outside loop
  - NaN recovery resets gradient state properly
  - constant-label skip removed (not needed with CrossEntropyLoss)
  - removed per-step debug logging noise
  - autocast uses correct API for torch 2.4
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from transformers import WhisperFeatureExtractor
from sklearn.metrics import roc_auc_score, average_precision_score

from ttm_dataset import TTMAudioDataset, ttm_collate_fn
from audio_model import WhisperTTM, build_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ── utils ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, scheduler, epoch, metric, cfg, tag="latest"):
    out_dir = Path(cfg["project"]["output_dir"]) / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch":     epoch,
        "metric":    metric,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config":    cfg,
    }
    path = out_dir / f"ckpt_{tag}.pt"
    torch.save(ckpt, path)
    logger.info("Checkpoint saved → %s  (metric=%.4f)", path, metric)
    return str(path)

def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    logger.info("Resumed from epoch %d  (metric=%.4f)", ckpt["epoch"], ckpt["metric"])
    return ckpt["epoch"]


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(all_logits: np.ndarray, all_labels: np.ndarray) -> Dict[str, float]:
    """all_logits: (N,2)  all_labels: (N,) int"""
    probs = torch.softmax(torch.from_numpy(all_logits.astype(np.float32)), dim=-1).numpy()[:, 1]

    # Replace NaN/inf
    if not np.isfinite(probs).all():
        n_bad = (~np.isfinite(probs)).sum()
        logger.warning("NaN/Inf in %d logits – replacing with 0.5", n_bad)
        probs = np.where(np.isfinite(probs), probs, 0.5)

    # Need both classes present for AUC
    if len(np.unique(all_labels)) < 2:
        acc = float(((probs > 0.5).astype(int) == all_labels).mean())
        return {"auc": 0.0, "ap": 0.0, "acc": acc}

    auc = roc_auc_score(all_labels, probs)
    ap  = average_precision_score(all_labels, probs)
    acc = float(((probs > 0.5).astype(int) == all_labels).mean())
    return {"auc": auc, "ap": ap, "acc": acc}


# ── one epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, cfg, device, epoch, split="train"):
    is_train    = split == "train"
    model.train() if is_train else model.eval()

    precision    = cfg["training"]["precision"]
    accum_steps  = cfg["training"]["accumulation_steps"]
    use_amp      = precision in ("bf16", "fp16")
    amp_dtype    = torch.bfloat16 if precision == "bf16" else torch.float16

    total_loss  = 0.0
    all_logits  = []
    all_labels  = []
    n_steps     = len(loader)
    nan_batches = 0

    if is_train:
        optimizer.zero_grad()

    for step, batch in enumerate(loader):
        features = batch["input_features"].to(device)   # (B, 80, 3000)
        labels   = batch["clip_labels"].to(device)      # (B,)  LongTensor

        # ── Forward ──────────────────────────────────────────────
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out    = model(features)
                logits = out["clip_logits"]              # (B, 2)  float
                loss   = criterion(logits, labels)
        else:
            out    = model(features)
            logits = out["clip_logits"]
            loss   = criterion(logits, labels)

        if is_train:
            loss = loss / accum_steps

        # ── NaN guard ────────────────────────────────────────────
        if not torch.isfinite(loss):
            nan_batches += 1
            logger.warning("NaN loss at step %d – resetting grads.", step)
            if optimizer is not None:
                optimizer.zero_grad()
            continue

        # ── Backward ─────────────────────────────────────────────
        if is_train:
            loss.backward()

        total_loss += loss.item() * (accum_steps if is_train else 1)
        all_logits.append(logits.detach().float().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        # ── Optimizer step ───────────────────────────────────────
        if is_train and ((step + 1) % accum_steps == 0 or step == n_steps - 1):
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()

        if is_train and step % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info("[%s e%02d %04d/%04d] loss=%.4f  lr=%.2e",
                        split, epoch, step, n_steps,
                        total_loss / max(step + 1, 1), lr)

    if nan_batches:
        logger.warning("Total NaN batches this epoch: %d / %d", nan_batches, n_steps)

    if not all_logits:
        logger.warning("No valid batches in epoch – returning zero metrics.")
        return {"loss": 0.0, "auc": 0.0, "ap": 0.0, "acc": 0.0}

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / max(len(loader), 1)

    # Save training metrics to train_history.json
    history_path = Path(cfg["project"]["output_dir"]) / "train_history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = {"train": [], "val": []}

    history[split].append({
        "epoch": epoch,
        "loss": metrics["loss"],
        "auc": metrics["auc"],
        "ap": metrics["ap"],
        "acc": metrics["acc"]
    })

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    logger.info("Metrics saved to %s", history_path)

    logger.info("[%s e%02d] loss=%.4f  auc=%.4f  ap=%.4f  acc=%.4f",
                split, epoch, metrics["loss"], metrics["auc"], metrics["ap"], metrics["acc"])
    return metrics


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s  |  VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    Path(cfg["project"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Feature extractor ────────────────────────────────────────────────────
    logger.info("Loading WhisperFeatureExtractor …")
    fe = WhisperFeatureExtractor.from_pretrained(cfg["model"]["backbone"])

    # ── Datasets ─────────────────────────────────────────────────────────────
    ann_dir = Path(cfg["data"]["annotations_dir"])

    train_ds = TTMAudioDataset(
        annotation_json=str(ann_dir / cfg["data"]["train_json"]),
        audio_root=cfg["data"]["root"],
        feature_extractor=fe,
        sample_rate=cfg["data"]["sample_rate"],
        max_audio_sec=cfg["data"]["max_audio_sec"],
        clip_overlap_sec=cfg["data"]["clip_overlap_sec"],
        split="train",
        audio_ext=cfg["data"]["audio_ext"],
        max_clips=cfg["data"].get("max_clips_train"),
    )
    val_ds = TTMAudioDataset(
        annotation_json=str(ann_dir / cfg["data"]["val_json"]),
        audio_root=cfg["data"]["root"],
        feature_extractor=fe,
        sample_rate=cfg["data"]["sample_rate"],
        max_audio_sec=cfg["data"]["max_audio_sec"],
        clip_overlap_sec=cfg["data"]["clip_overlap_sec"],
        split="val",
        audio_ext=cfg["data"]["audio_ext"],
        max_clips=cfg["data"].get("max_clips_val"),
    )

    logger.info("Train windows: %d  |  Val windows: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["training"]["dataloader_workers"],
        pin_memory=cfg["training"]["pin_memory"],
        collate_fn=ttm_collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False, num_workers=cfg["training"]["dataloader_workers"],
        pin_memory=cfg["training"]["pin_memory"],
        collate_fn=ttm_collate_fn,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = WhisperTTM(
        model_name=cfg["model"]["backbone"],
        freeze_encoder_layers=cfg["model"]["freeze_encoder_layers"],
        projection_dim=cfg["model"]["projection_dim"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
    ).to(device).float()   # force all weights to fp32

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Params – trainable: %s  total: %s",
                f"{trainable:,}", f"{total:,}")

    # ── Optimiser / scheduler ────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps  = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = max(1, int(total_steps * cfg["training"]["warmup_ratio"]))

    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                             total_iters=warmup_steps)
    cosine_sched = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, (total_steps - warmup_steps) // cfg["training"]["num_cycles"]),
    )
    scheduler = SequentialLR(optimizer,
                              schedulers=[warmup_sched, cosine_sched],
                              milestones=[warmup_steps])

    criterion = build_loss(cfg)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_metric  = 0.0
    patience_ctr = 0

    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    history = []

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["training"]["epochs"]):

        train_metrics = run_epoch(model, train_loader, criterion,
                                  optimizer, cfg, device, epoch, "train")
        scheduler.step()

        val_metrics = run_epoch(model, val_loader, criterion,
                                None, cfg, device, epoch, "val")

        monitor = val_metrics.get("auc", 0.0)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if (epoch + 1) % cfg["training"]["save_every_n_epochs"] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, monitor, cfg, "latest")

        if monitor > best_metric:
            best_metric  = monitor
            patience_ctr = 0
            save_checkpoint(model, optimizer, scheduler, epoch, monitor, cfg, "best")
            logger.info("★ New best  val_auc=%.4f", best_metric)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["training"]["early_stopping_patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

    out_dir = Path(cfg["project"]["output_dir"])
    hist_path = out_dir / "train_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Done. History → %s", hist_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--resume",  default=None)
    main(parser.parse_args())