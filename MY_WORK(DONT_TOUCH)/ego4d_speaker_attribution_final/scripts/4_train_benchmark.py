#!/usr/bin/env python3

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.dataset.ttm_benchmark_dataset import TTMBenchmarkDataset, ttm_benchmark_collate
from src.models import build_benchmark_model
from src.utils.config import load_config
from src.utils.logger import setup_logger


def build_dataset(config, split_key):
    benchmark = config["benchmark"]
    return TTMBenchmarkDataset(
        split_file=benchmark[split_key],
        video_dir=benchmark["video_dir"],
        audio_dir=benchmark["audio_dir"],
        tracklet_dir=benchmark["tracklet_dir"],
        ttm_dir=benchmark["ttm_dir"],
        fps=benchmark["fps"],
        sample_rate=benchmark["sample_rate"],
        crop_size=benchmark["crop_size"],
        min_frames=benchmark["min_frames"],
        max_frames=benchmark["max_frames"],
        frame_step=benchmark["frame_step"],
        min_valid_face_frames=benchmark["min_valid_face_frames"],
        filter_missing_tracks=benchmark["filter_missing_tracks"],
    )


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def forward_model(model, model_type, batch):
    if model_type == "audio_only":
        return model(audio=batch["audio"], audio_mask=batch["audio_mask"])
    if model_type == "visual_only":
        return model(video=batch["video"], face_mask=batch["face_mask"], frame_mask=batch["frame_mask"])
    return model(
        video=batch["video"],
        audio=batch["audio"],
        face_mask=batch["face_mask"],
        frame_mask=batch["frame_mask"],
        audio_mask=batch["audio_mask"],
    )


def evaluate(model, loader, device, model_type):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = forward_model(model, model_type, batch)
            labels = batch["label"]
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    parser.add_argument(
        "--model_type",
        default="audio_only",
        choices=["audio_only", "visual_only", "fusion_cross_attention"],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "models", "benchmark"))
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(f"benchmark_train_{args.model_type}", config["logging"]["log_dir"])

    device_name = config["training"]["device"]
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")

    train_dataset = build_dataset(config, "train_split")
    val_dataset = build_dataset(config, "val_split")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ttm_benchmark_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ttm_benchmark_collate,
    )

    model = build_benchmark_model(
        args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    pos_weight = torch.tensor([config["training"].get("pos_weight", 1.0)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = -1.0
    best_path = os.path.join(args.output_dir, f"{args.model_type}_best.pt")

    logger.info("train_segments=%d val_segments=%d", len(train_dataset), len(val_dataset))
    logger.info("training model_type=%s on %s", args.model_type, device)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            batch = move_batch_to_device(batch, device)
            logits = forward_model(model, args.model_type, batch)
            labels = batch["label"]

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_count += labels.size(0)

        train_loss = running_loss / max(running_count, 1)
        metrics = evaluate(model, val_loader, device, args.model_type)
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1,
            train_loss,
            metrics["loss"],
            metrics["acc"],
        )

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            torch.save(
                {
                    "model_type": args.model_type,
                    "state_dict": model.state_dict(),
                    "val_acc": best_acc,
                    "args": vars(args),
                },
                best_path,
            )
            logger.info("saved best checkpoint to %s", best_path)


if __name__ == "__main__":
    main()
