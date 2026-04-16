"""
Visual Pipeline - run.py  (ViT Edition)
========================================
Full training entry point with:
  - Video Swin V2 / TimeSformer / Factorised ViT
  - Cosine annealing LR with linear warmup (via transformers or manual)
  - EMA model weights
  - MixUp data augmentation
  - torch.compile (PyTorch 2.x)
  - Automatic AMP dtype (bfloat16 on Ampere, float16 on older)
  - Full checkpoint save/resume with optimizer + scheduler state
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import ViTVisualConfig, get_loader, mixup_batch
from model import build_model
from engine import (
    ModelEMA, WeightedBinaryCrossEntropyLoss,
    get_amp_dtype,
    train, validate, infer,
    save_checkpoint, load_checkpoint,
)


# ──────────────────────────────────────────────────────────────
#  Cosine LR with linear warmup  (no extra dependency)
# ──────────────────────────────────────────────────────────────

class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer  : torch.optim.Optimizer,
        warmup_steps: int,
        total_steps : int,
        min_lr_ratio: float = 0.1,
    ):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(warmup_steps, 1)
            progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
            import math
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine)

        super().__init__(optimizer, lr_lambda, last_epoch=-1)


# ──────────────────────────────────────────────────────────────
#  Args
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ViT Visual TTM Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── paths ──
    g = p.add_argument_group("Paths")
    g.add_argument("--source_path", required=True,
                   help="Root dir of extracted video frames per UID")
    g.add_argument("--json_path",   required=True,
                   help="Face tracklet JSONs (json_original/)")
    g.add_argument("--gt_path",     required=True,
                   help="Ground-truth LAM JSONs (result_LAM/)")
    g.add_argument("--train_file",  required=True, help="train.list")
    g.add_argument("--val_file",    required=True, help="val.list")
    g.add_argument("--test_path",   default="",    help="Test/infer dir (eval only)")
    g.add_argument("--exp_path",    default="experiments/vit_ttm",
                   help="Checkpoint + log output dir")
    g.add_argument("--checkpoint",  default=None,  help="Resume from .pth")

    # ── model ──
    g = p.add_argument_group("Model")
    g.add_argument("--model", default="VideoSwinV2TTM",
                   choices=["VideoSwinV2TTM", "TimeSformerTTM", "FactorisedViTTTM", "DinoViTTTM", "DinoViTTrackTTM"],
                   help="Model architecture")
    g.add_argument("--use_track",        action="store_true", default=True,
                   help="Use spatial track features from json_original")
    g.add_argument("--freeze_backbone",  action="store_true", default=True,
                   help="Freeze ViT backbone (for DINO models)")
    g.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone")
    g.add_argument("--temporal_depth",   type=int, default=2,
                   help="Number of temporal transformer layers")
    g.add_argument("--variant", default="swin3d_s",
                   help="Backbone variant: swin3d_s | swin3d_b | vit_small_patch16_224 | ...")
    g.add_argument("--clip_frames",    type=int,   default=8,
                   help="Frames per clip (T). Use 8 or 16 for ViT tube tokens")
    g.add_argument("--img_size",       type=int,   default=224)
    g.add_argument("--lstm_hidden",    type=int,   default=256,
                   help="(Unused for ViT models — kept for CLI compat)")
    g.add_argument("--dropout",        type=float, default=0.2)
    g.add_argument("--freeze_stages",  type=int,   default=0,
                   help="Freeze first N stages of Swin backbone (0 = train all)")

    # ── data ──
    g = p.add_argument_group("Data")
    g.add_argument("--train_stride", type=int, default=4,
                   help="Frame subsampling — ViT wants dense clips; keep ≤ 8")
    g.add_argument("--val_stride",   type=int, default=4)
    g.add_argument("--test_stride",  type=int, default=1)
    g.add_argument("--batch_size",   type=int, default=16,
                   help="Per-GPU batch size (ViT uses more VRAM than CNN)")
    g.add_argument("--num_workers",  type=int, default=4)
    g.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Accumulate gradients over this many mini-batches to increase effective batch size")
    g.add_argument("--mixup",        action="store_true", default=False)
    g.add_argument("--no_mixup",     action="store_false", dest="mixup")
    g.add_argument("--mixup_alpha",  type=float, default=0.2)

    # ── training ──
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",        type=int,   default=30)
    g.add_argument("--lr",            type=float, default=1e-4,
                   help="Peak LR. ViT needs lower LR than CNN (1e-4 to 5e-5)")
    g.add_argument("--backbone_lr_scale", type=float, default=0.1,
                   help="Scale backbone LR by this factor (layer-wise decay)")
    g.add_argument("--weight_decay",  type=float, default=0.05,
                   help="AdamW weight decay (higher for ViT: 0.05)")
    g.add_argument("--warmup_epochs", type=int,   default=3)
    g.add_argument("--grad_clip",     type=float, default=1.0)
    g.add_argument("--weights",       type=float, nargs=2,
                   default=[1.0, 1.0],
                   help="Class weights [neg_weight, pos_weight]")
    g.add_argument("--label_smoothing", type=float, default=0.0)
    g.add_argument("--ema",           action="store_true", default=True)
    g.add_argument("--no_ema",        action="store_false", dest="ema")
    g.add_argument("--ema_decay",     type=float, default=0.9998)
    g.add_argument("--early_stop_patience", type=int, default=6,
                   help="Stop after this many epochs without meaningful mAP improvement (<=0 disables)")
    g.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                   help="Minimum mAP gain to reset early-stopping counter")
    g.add_argument("--amp",           action="store_true", default=True)
    g.add_argument("--no_amp",        action="store_false", dest="amp")
    g.add_argument("--compile",       action="store_true",
                   help="torch.compile (PyTorch 2.x, requires triton)")

    # ── mode ──
    g = p.add_argument_group("Mode")
    g.add_argument("--eval",  action="store_true", help="Evaluation only")
    g.add_argument("--infer", action="store_true", help="Unannotated inference")

    # ── device ──
    g = p.add_argument_group("Device")
    g.add_argument("--device_id", type=int, default=0)
    g.add_argument("--seed",      type=int, default=42)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────
#  Layer-wise LR decay  (important for ViT fine-tuning)
# ──────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, args) -> optim.AdamW:
    backbone_params, head_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Use explicit module prefix split: everything under backbone uses scaled LR,
        # everything else (temporal/fusion/classifier heads) uses base LR.
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    backbone_lr = args.lr * args.backbone_lr_scale
    head_lr = args.lr

    print(f"[Optimizer] backbone={len(backbone_params)} params  "
          f"head={len(head_params)} params")
    print(f"[Optimizer] lr_backbone={backbone_lr:.2e}  lr_head={head_lr:.2e}")

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})

    return optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── reproducibility ──
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # ── device ──
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = torch.device("cpu")
        print("[Warning] No CUDA — running on CPU")

    amp_dtype = get_amp_dtype(device)
    print(f"[AMP] dtype = {amp_dtype}")

    os.makedirs(args.exp_path, exist_ok=True)
    with open(os.path.join(args.exp_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── model ──
    print(f"\n[Model] {args.model}  variant={args.variant}")
    model_kwargs = dict(pretrained=True, num_classes=2, dropout=args.dropout)

    if args.model == "VideoSwinV2TTM":
        model_kwargs["variant"]         = args.variant
        model_kwargs["freeze_stages"]   = args.freeze_stages
    elif args.model in ("TimeSformerTTM", "FactorisedViTTTM"):
        model_kwargs["vit_variant"]     = args.variant
        model_kwargs["num_frames"]      = args.clip_frames
    elif args.model in ("DinoViTTTM", "DinoViTTrackTTM"):
        model_kwargs["vit_variant"]     = args.variant
        model_kwargs["num_frames"]      = args.clip_frames
        model_kwargs["temporal_depth"]  = args.temporal_depth
        model_kwargs["freeze_backbone"] = args.freeze_backbone

    model = build_model(args.model, **model_kwargs).to(device)

    # ── torch.compile ──
    if args.compile and hasattr(torch, "compile"):
        print("[torch.compile] Compiling model …")
        model = torch.compile(model, mode="reduce-overhead")

    # ── EMA ──
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None
    if ema:
        print(f"[EMA] decay={args.ema_decay}")

    # ── optimizer ──
    optimizer = build_optimizer(model, args)

    # ── resume ──
    start_epoch = 0
    best_mAP    = 0.0
    if args.checkpoint:
        model, start_epoch, best_mAP = load_checkpoint(
            model, args.checkpoint, optimizer, device=device
        )
        ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None
        start_epoch += 1

    # ── data config ──
    cfg = ViTVisualConfig(
        source_path  = args.source_path,
        json_path    = args.json_path,
        gt_path      = args.gt_path,
        train_file   = args.train_file,
        val_file     = args.val_file,
        test_path    = args.test_path,
        clip_frames  = args.clip_frames,
        img_size     = args.img_size,
        train_stride = args.train_stride,
        val_stride   = args.val_stride,
        test_stride  = args.test_stride,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        use_track    = args.use_track,
        # use_mixup    = args.mixup,
        # mixup_alpha  = args.mixup_alpha,
    )

    # ── eval / infer ──
    if args.eval:
        mode   = "infer" if args.infer else "test"
        loader = get_loader(cfg, mode)
        eval_model = ema.model if ema else model
        # test/infer loaders do not provide GT labels; write predictions instead.
        infer(loader, eval_model, args.exp_path, device, amp_dtype)
        return

    # ── dataloaders ──
    train_loader = get_loader(cfg, "train")
    val_loader   = get_loader(cfg, "val")

    # ── loss ──
    class_weights = torch.FloatTensor(args.weights).to(device)
    criterion = WeightedBinaryCrossEntropyLoss(
        weight          = class_weights,
        label_smoothing = args.label_smoothing,
    )

    # ── scheduler ──
    update_steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum_steps, 1))
    total_steps  = args.epochs * update_steps_per_epoch
    warmup_steps = args.warmup_epochs * update_steps_per_epoch
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps = warmup_steps,
        total_steps  = total_steps,
        min_lr_ratio = 0.1,
    )

    # ── AMP scaler (only for float16; bfloat16 doesn't need it) ──
    scaler = (
        GradScaler(device="cuda")
        if args.amp and device.type == "cuda" and amp_dtype == torch.float16
        else None
    )

    # ── MixUp function ──
    mixup_fn = None
    if args.mixup:
        from functools import partial
        mixup_fn = partial(mixup_batch, alpha=args.mixup_alpha, num_classes=2)

    # ──────────────────────────────────────────────────────────
    #  Training loop
    # ──────────────────────────────────────────────────────────

    print(f"\n{'─'*60}")
    print(f"  Model      : {args.model} ({args.variant})")
    print(f"  Device     : {device}  AMP={args.amp} ({amp_dtype})")
    print(f"  Epochs     : {start_epoch} → {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Grad accum : {args.grad_accum_steps} (effective batch ~ {args.batch_size * args.grad_accum_steps})")
    print(f"  Clip       : {args.clip_frames} frames @ {args.img_size}px")
    print(f"  EMA        : {ema is not None}")
    print(f"  MixUp      : {args.mixup}")
    print(f"  Best mAP   : {best_mAP:.4f}")
    if args.early_stop_patience > 0:
        print(f"  Early stop : patience={args.early_stop_patience}  min_delta={args.early_stop_min_delta}")
    else:
        print("  Early stop : disabled")
    print(f"{'─'*60}\n")

    no_improve_epochs = 0

    for epoch in range(start_epoch, args.epochs):

        train_loss = train(
            loader       = train_loader,
            model        = model,
            criterion    = criterion,
            optimizer    = optimizer,
            epoch        = epoch,
            scaler       = scaler,
            ema          = ema,
            device       = device,
            amp_dtype    = amp_dtype,
            mixup_fn     = mixup_fn,
            grad_clip    = args.grad_clip,
            grad_accum_steps = args.grad_accum_steps,
            scheduler=scheduler,
        )

        #scheduler.step()

        # validate with EMA model if available
        val_model = ema.model if ema else model
        mAP, auc = validate(val_loader, val_model, device, amp_dtype, "val", criterion)

        improved = (mAP - best_mAP) > args.early_stop_min_delta
        is_best  = improved
        if improved:
            best_mAP = mAP
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        ckpt_state = {
            "epoch"      : epoch,
            "state_dict" : model.state_dict(),
            "ema"        : ema.state_dict() if ema else None,
            "optimizer"  : optimizer.state_dict(),
            "scheduler"  : scheduler.state_dict(),
            "mAP"        : mAP,
            "auc"        : auc,
            "args"       : vars(args),
        }
        save_checkpoint(
            ckpt_state,
            save_path = args.exp_path,
            is_best   = is_best,
            filename  = f"epoch_{epoch:03d}.pth",
        )

        lr_now = optimizer.param_groups[-1]["lr"]
        tag    = " ← BEST" if is_best else ""
        print(f"[Epoch {epoch:3d}] loss={train_loss:.4f}  "
              f"mAP={mAP:.4f}  auc={auc:.4f}  best={best_mAP:.4f}  "
              f"lr={lr_now:.2e}{tag}\n")

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(f"[Early Stop] No meaningful mAP improvement for {no_improve_epochs} epochs. Stopping at epoch {epoch}.")
            break

    print(f"\n{'─'*60}")
    print(f"  Training complete")
    print(f"  Best mAP = {best_mAP:.4f}")
    print(f"  Best model → {args.exp_path}/best_model.pth")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
