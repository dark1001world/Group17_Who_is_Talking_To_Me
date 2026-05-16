import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

# Import your updated loader and the Whisper model
from data_loader import AudioConfig, get_audio_loader
from model import WhisperTTM 

# Import your existing engine functions
from engine import (
    train, validate, ModelEMA, 
    WeightedBinaryCrossEntropyLoss, 
    save_checkpoint, load_checkpoint,
    get_amp_dtype
)

# ──────────────────────────────────────────────────────────────
#  Cosine LR with linear warmup
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
#  Main Entry Point
# ──────────────────────────────────────────────────────────────
def main():
    # 1. Configuration
    # Note: val_pos_stride=4 ensures we match the 13,220 visual samples exactly
    cfg = AudioConfig(
        wave_path="/DATA/G17/Data/wave/",
        train_annotations="/DATA/G17/Data/extract_data/ttm_train_data.json",
        val_annotations="/DATA/G17/Data/extract_data/ttm_validation_data.json",
        batch_size=48, 
        num_workers=12,
        pos_stride=4,
        neg_stride=60,
        val_pos_stride=4, 
        val_neg_stride=60
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(device)
    print(f"Using device: {device} | AMP Dtype: {amp_dtype}")

    # 2. DataLoaders
    train_loader = get_audio_loader(cfg, mode="train")
    val_loader   = get_audio_loader(cfg, mode="val")

    # 3. Model
    model = WhisperTTM(
        model_size="base", 
        freeze_backbone=True, 
        temporal_depth=2,
        dropout=0.3 # Increased to prevent overfitting
    ).to(device)

    # 4. Training Utilities
    criterion = WeightedBinaryCrossEntropyLoss(weight=torch.tensor([1.0, 1.2]))
    
    # Filter for trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Initialize Optimizer (Fine-tuning settings)
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.05)
    
    scaler = GradScaler()
    ema = ModelEMA(model, decay=0.999)
    
    # 5. Resume Logic
    resume_path = "experiments/whisper_audio_baseline/best_model.pth" 
    start_epoch = 0
    best_map = 0.0

    if resume_path and os.path.exists(resume_path):
        print(f"\nResuming training from {resume_path}...")
        model, start_epoch, best_map = load_checkpoint(
            model=model, 
            ckpt_path=resume_path, 
            optimizer=optimizer, 
            device=device
        )
        
        # Restore EMA weights
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        if "ema_state_dict" in ckpt:
            ema.model.load_state_dict(ckpt["ema_state_dict"])
            print("  ✓ EMA state restored")
        elif "ema" in ckpt and ckpt["ema"] is not None:
             ema.model.load_state_dict(ckpt["ema"])
             print("  ✓ EMA state restored from dict")

        # --- MANDATORY: Override restored checkpoint state ---
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
            param_group['weight_decay'] = 0.05
        print(f"  ✓ FORCED LR to 1e-5 and WD to 0.05 (Overrode checkpoint)")    
    else:
        print("\nStarting fresh Whisper-TTM training run...")

    # 6. Scheduler Setup
    epochs = 15
    exp_path = "experiments/whisper_audio_baseline" 
    os.makedirs(exp_path, exist_ok=True)
    
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch  
    warmup_steps = 0 # Starting from 61%, no warmup needed
    
    scheduler = CosineWarmupScheduler(
        optimizer, 
        warmup_steps=warmup_steps, 
        total_steps=total_steps
    )
    
    # Fast-forward scheduler to the current epoch
    if start_epoch > 0:
        completed_steps = start_epoch * steps_per_epoch
        for _ in range(completed_steps):
            scheduler.step()
        print(f"  ✓ Scheduler caught up to step {completed_steps}")

    # 7. Training Loop
    print("\nStarting Whisper Audio Training...")
    for epoch in range(start_epoch + 1, epochs + 1):
        # Train
        train_loss = train(
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            scaler=scaler,
            ema=ema,
            device=device,
            amp_dtype=amp_dtype,
            log_interval=50,
            scheduler=scheduler # Batch-level decay handled by engine
        )

        # Validate
        print(f"\n--- Epoch {epoch} Validation ---")
        val_map, val_auc = validate(
            loader=val_loader,
            model=ema.model, 
            device=device,
            mode="val",
            criterion=criterion,
            amp_dtype=amp_dtype
        )

        # Save Checkpoint
        is_best = val_map > best_map
        if is_best:
            best_map = val_map
            
        save_checkpoint(
            state={
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "mAP": val_map,
                "optimizer": optimizer.state_dict(),
            },
            save_path=exp_path,
            is_best=is_best,
            filename=f"checkpoint_ep{epoch:02d}.pth"
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} Complete. LR: {current_lr:.2e} | Best mAP: {best_map:.4f}\n")

if __name__ == "__main__":
    main()