#!/usr/bin/env python3

import argparse
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.dataset.ttm_benchmark_dataset import TTMBenchmarkDataset, ttm_benchmark_collate
from src.models import build_benchmark_model
from src.utils.config import load_config


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    parser.add_argument(
        "--model_type",
        required=True,
        choices=["audio_only", "visual_only", "fusion_cross_attention"],
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    benchmark = config["benchmark"]
    split_key = "train_split" if args.split == "train" else "val_split"
    dataset = TTMBenchmarkDataset(
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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ttm_benchmark_collate)

    device_name = config["training"]["device"]
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})
    model = build_benchmark_model(
        args.model_type,
        hidden_dim=model_args.get("hidden_dim", 256),
        num_layers=model_args.get("num_layers", 2),
        dropout=model_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    grouped = defaultdict(list)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = forward_model(model, args.model_type, batch)
            probs = torch.sigmoid(logits).cpu()
            labels = batch["label"].cpu()

            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for meta, prob, label in zip(batch["metadata"], probs.tolist(), labels.tolist()):
                key = (meta["clip_uid"], meta["start_frame"], meta["end_frame"], meta["segment_index"])
                grouped[key].append(
                    {
                        "person_id": meta["person_id"],
                        "score": float(prob),
                        "label": float(label),
                    }
                )

    print("sample_acc:", correct / max(total, 1))
    print("candidate_decisions:")
    shown = 0
    for key, candidates in grouped.items():
        best = max(candidates, key=lambda item: item["score"])
        print(
            {
                "clip_uid": key[0],
                "start_frame": key[1],
                "end_frame": key[2],
                "predicted_person_id": best["person_id"],
                "score": round(best["score"], 4),
                "num_candidates": len(candidates),
            }
        )
        shown += 1
        if shown >= 10:
            break


if __name__ == "__main__":
    main()
