import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import the fusion components we just built
from dataset import FusionDataset
from model import CrossModalTemporalFusion

# Import your robust engine tools
from engine import (
    train, validate, ModelEMA, 
    WeightedBinaryCrossEntropyLoss, 
    save_checkpoint, get_amp_dtype
)

# Reuse your scheduler
class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / max(warmup_steps, 1)
            progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
            import math
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        super().__init__(optimizer, lr_lambda, last_epoch=-1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(device)
    print(f"Fusion Training on {device} with {amp_dtype}")

    # 1. Paths
    AUDIO_TRAIN = "/DATA/G17/Data/extracted_features/audio/train"
    AUDIO_VAL   = "/DATA/G17/Data/extracted_features/audio/val"
    VISUAL_TRAIN = "/DATA/G17/Data/extracted_features/visual/train"
    VISUAL_VAL   = "/DATA/G17/Data/extracted_features/visual/val"
    
    TRAIN_JSON = "/DATA/G17/Data/extract_data/ttm_train_data.json"
    VAL_JSON   = "/DATA/G17/Data/extract_data/ttm_validation_data.json"

    # 2. Datasets & Loaders
    # Since inputs are just small vectors, we can use a massive batch size!
    BATCH_SIZE = 512 
    
    print("\nLoading Training Set...")
    train_dataset = FusionDataset(AUDIO_TRAIN, VISUAL_TRAIN, TRAIN_JSON)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=8, pin_memory=True, drop_last=True
    )

    print("\nLoading Validation Set...")
    val_dataset = FusionDataset(AUDIO_VAL, VISUAL_VAL, VAL_JSON)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=8, pin_memory=True
    )

    # 3. Model
    # visual_dim is 768 (DINO), audio_dim is 512 (Whisper)
    model = CrossModalTemporalFusion(
        audio_dim=512, visual_dim=768, shared_dim=512, 
        num_heads=8, dropout=0.3
    ).to(device)

    # 4. Training Setup
    # Because this model trains from scratch (no pretrained weights), 
    # we can use a higher learning rate: 5e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
    criterion = WeightedBinaryCrossEntropyLoss(weight=torch.tensor([1.0, 1.2]))
    
    scaler = GradScaler()
    ema = ModelEMA(model, decay=0.999)
    
    # Scheduler: Let's warm up for 2 epochs, then cosine decay
    epochs = 20
    steps_per_epoch = len(train_loader)
    scheduler = CosineWarmupScheduler(
        optimizer, 
        warmup_steps=2 * steps_per_epoch, 
        total_steps=epochs * steps_per_epoch
    )

    exp_path = "experiments/ttm_final_fusion2"
    os.makedirs(exp_path, exist_ok=True)
    best_map = 0.0

    # 5. Training Loop
    print("\nStarting Cross-Modal Fusion Training...")
    for epoch in range(1, epochs + 1):
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
            log_interval=20,
            scheduler=scheduler
        )

        print(f"\n--- Epoch {epoch} Validation ---")
        val_map, val_auc = validate(
            loader=val_loader,
            model=ema.model, 
            device=device,
            mode="val",
            criterion=criterion,
            amp_dtype=amp_dtype
        )

        is_best = val_map > best_map
        if is_best:
            best_map = val_map
            
        save_checkpoint(
            state={
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "mAP": val_map,
                "optimizer": optimizer.state_dict()
            },
            save_path=exp_path,
            is_best=is_best,
            filename=f"checkpoint_ep{epoch:02d}.pth"
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} Complete. LR: {current_lr:.2e} | Best Fusion mAP: {best_map:.4f}\n")

if __name__ == "__main__":
    main()