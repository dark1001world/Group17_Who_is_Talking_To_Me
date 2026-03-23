"""
Training script for Audio Encoder Pipeline.

This script trains a frame-level audio encoder on the Ego4D TTM task.
It includes:
- Loading data with custom DataLoader
- Initializing HuBERT encoder
- Training loop with AdamW optimizer
- Cosine annealing scheduler
- Loss computation and logging
- Model checkpointing

Usage:
    python train_audio_encoder.py \
        --annotation-file /path/to/annotations.json \
        --audio-root /path/to/audio \
        --output-dir ./checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import time

from audio_encoder import AudioEncoder
from config import AudioConfig, TrainingConfig, DataConfig, get_config
from dataset import create_dataloaders, Ego4D_TTM_Dataset
from utils import (
    DeviceUtils,
    PaddingUtils,
    LossFunctions,
    MetricsUtils,
)


class TTM_Classifier(nn.Module):
    """
    Simple TTM (Talking To Me) classifier.

    Takes audio embeddings and outputs frame-level predictions.
    """

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256):
        """
        Initialize classifier.

        Args:
            embedding_dim (int): Audio embedding dimension.
            hidden_dim (int): Hidden layer dimension.
        """
        super().__init__()

        self.audio_encoder = None  # Will be set after initialization

        # Frame-level classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),  # Binary output
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio (torch.Tensor): Raw audio. Shape: (batch, num_samples)
            audio_mask (torch.Tensor): Attention mask. Shape: (batch, num_samples)

        Returns:
            logits (torch.Tensor): Frame-level predictions. Shape: (batch, num_frames)
        """
        # Get embeddings: (batch, num_frames, embedding_dim)
        embeddings, _ = self.audio_encoder(audio, audio_mask)

        # Apply classifier to each frame: (batch, num_frames, 1)
        logits = self.classifier(embeddings)

        # Squeeze last dimension: (batch, num_frames)
        logits = logits.squeeze(-1)

        return logits


class Trainer:
    """Training manager for audio encoder."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        device: torch.device,
        output_dir: str = "./checkpoints",
    ):
        """
        Initialize trainer.

        Args:
            model (nn.Module): Model to train.
            optimizer (optim.Optimizer): Optimizer.
            scheduler (lr_scheduler): Learning rate scheduler.
            device (torch.device): Device to train on.
            output_dir (str): Directory for saving checkpoints.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.writer = SummaryWriter(str(self.output_dir / "logs"))
        self.global_step = 0

    def train_epoch(
        self,
        train_loader,
        epoch: int,
        config: TrainingConfig,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training DataLoader.
            epoch (int): Epoch number.
            config (TrainingConfig): Training config.

        Returns:
            metrics (dict): Training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            audio = batch["audio"].to(self.device)
            audio_mask = batch["audio_mask"].to(self.device)
            labels_list = batch["labels"]

            # Forward pass
            self.optimizer.zero_grad()

            logits = self.model(audio, audio_mask)

            # Compute loss
            loss = self._compute_batch_loss(
                logits, labels_list, audio_mask, config.device
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=config.max_grad_norm
            )
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": loss.item()})

            # Save checkpoint periodically
            if self.global_step % config.save_every_n_steps == 0:
                self.save_checkpoint(epoch, batch_idx)

        avg_loss = total_loss / num_batches
        self.writer.add_scalar("train/loss", avg_loss, epoch)

        return {"loss": avg_loss}

    def validate(
        self,
        val_loader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation DataLoader.
            epoch (int): Epoch number.

        Returns:
            metrics (dict): Validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")

            for batch in pbar:
                audio = batch["audio"].to(self.device)
                audio_mask = batch["audio_mask"].to(self.device)
                labels_list = batch["labels"]

                # Forward
                logits = self.model(audio, audio_mask)

                # Loss
                loss = self._compute_batch_loss(
                    logits, labels_list, audio_mask, self.device
                )

                total_loss += loss.item()
                num_batches += 1

                # Collect predictions for metrics
                for i, (logit, label) in enumerate(zip(logits, labels_list)):
                    label = label.to(self.device)
                    # Truncate to match frame count
                    num_frames = min(logit.shape[0], label.shape[0])
                    all_preds.append(logit[:num_frames])
                    all_targets.append(label[:num_frames])

        avg_loss = total_loss / num_batches

        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = MetricsUtils.compute_metrics(all_preds, all_targets)

        # Log
        self.writer.add_scalar("val/loss", avg_loss, epoch)
        self.writer.add_scalar("val/accuracy", metrics["accuracy"], epoch)
        self.writer.add_scalar("val/f1", metrics["f1"], epoch)

        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }

    def _compute_batch_loss(
        self,
        logits: torch.Tensor,
        labels_list: list,
        audio_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute loss for a batch (with variable-length labels).

        Args:
            logits (torch.Tensor): Model predictions. Shape: (batch, num_frames)
            labels_list (list): List of labels (variable length per sample).
            audio_mask (torch.Tensor): Audio mask. Shape: (batch, num_samples)
            device (torch.device): Device.

        Returns:
            loss (torch.Tensor): Scalar loss.
        """
        total_loss = 0.0
        num_samples = 0

        # HuBERT reduction factor (4x)
        hubert_reduction = 4
        audio_frames = audio_mask.shape[1] // hubert_reduction

        for i, (logit, label) in enumerate(zip(logits, labels_list)):
            label = label.to(device)

            # Align lengths
            num_frames = min(logit.shape[0], label.shape[0], audio_frames)
            logit_aligned = logit[:num_frames]
            label_aligned = label[:num_frames]

            # Compute loss
            batch_loss = LossFunctions.frame_level_bce_loss(
                logit_aligned.unsqueeze(0),
                label_aligned.unsqueeze(0),
                pos_weight=1.0,
            )

            total_loss += batch_loss
            num_samples += 1

        return total_loss / num_samples

    def save_checkpoint(self, epoch: int, batch_idx: int):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }

        save_path = (
            self.output_dir / f"checkpoint_epoch{epoch}_step{self.global_step}.pt"
        )
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")

    def save_final_model(self, epoch: int):
        """Save final trained model."""
        save_path = self.output_dir / f"final_model_epoch{epoch}.pt"
        torch.save(self.model.state_dict(), save_path)
        print(f"Final model saved: {save_path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train audio encoder for Ego4D TTM task"
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        required=False,
        default="./data/annotations.json",
        help="Path to annotation JSON file",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        required=False,
        default="./data/audio",
        help="Root directory for audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to train on (auto-detect if not specified)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )

    args = parser.parse_args()

    # Get configs
    audio_config, train_config, data_config = get_config()

    # Override with command-line args
    train_config.num_epochs = args.num_epochs
    train_config.batch_size = args.batch_size
    train_config.learning_rate = args.learning_rate

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = DeviceUtils.get_device(prefer_cuda=True)

    print(f"Training on device: {device}")
    print(f"Audio config: {audio_config}")
    print(f"Training config: {train_config}")

    # Create encoder and classifier
    print("\nInitializing audio encoder...")
    audio_encoder = AudioEncoder(audio_config)
    audio_encoder.to(device)

    print("Initializing TTM classifier...")
    classifier = TTM_Classifier(embedding_dim=audio_config.embedding_dim)
    classifier.audio_encoder = audio_encoder
    classifier.to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        classifier.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    total_steps = train_config.num_epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=train_config.learning_rate * 0.1,
    )

    # Create trainer
    trainer = Trainer(classifier, optimizer, scheduler, device, args.output_dir)

    # Note: Data loading example (requires actual data)
    # In practice, you would:
    # train_loader, val_loader, test_loader = create_dataloaders(
    #     args.annotation_file,
    #     args.audio_root,
    #     audio_config,
    #     train_batch_size=train_config.batch_size,
    # )

    print("\n" + "=" * 60)
    print("TRAINING SCRIPT INITIALIZED")
    print("=" * 60)
    print(f"Models saved to: {args.output_dir}")
    print("\nTo train with data:")
    print("1. Prepare annotation JSON file with video segments")
    print("2. Ensure audio files are in --audio-root directory")
    print("3. Uncomment DataLoader creation in main()")
    print("4. Run: python train_audio_encoder.py --annotation-file ... --audio-root ...")
    print("=" * 60)

    # Example training loop (commented out - requires actual data)
    """
    best_val_loss = float("inf")

    for epoch in range(train_config.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{train_config.num_epochs}")
        print('='*60)

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch, train_config)
        print(f"Train Loss: {train_metrics['loss']:.4f}")

        # Validate
        val_metrics = trainer.validate(val_loader, epoch)
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")

        # Learning rate scheduling
        scheduler.step()

        # Track best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            trainer.save_final_model(epoch)

    # Test
    print(f"\n{'='*60}")
    print("Testing")
    print('='*60)
    test_metrics = trainer.validate(test_loader, train_config.num_epochs)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    """


if __name__ == "__main__":
    main()
