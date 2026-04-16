#!/usr/bin/env python3

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.dataset.ttm_benchmark_dataset import TTMBenchmarkDataset
from src.utils.config import load_config


def describe_sample(sample):
    meta = sample["metadata"]
    print("clip_uid:", meta["clip_uid"])
    print("person_id:", meta["person_id"])
    print("segment_index:", meta["segment_index"])
    print("frame_range:", (meta["start_frame"], meta["end_frame"]))
    print("label:", float(sample["label"].item()))
    print("video_shape:", tuple(sample["video"].shape))
    print("audio_shape:", tuple(sample["audio"].shape))
    print("valid_face_frames:", meta["valid_face_frames"])


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    benchmark = config["benchmark"]

    train_dataset = TTMBenchmarkDataset(
        split_file=benchmark["train_split"],
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

    val_dataset = TTMBenchmarkDataset(
        split_file=benchmark["val_split"],
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

    print("train_segments:", len(train_dataset))
    print("val_segments:", len(val_dataset))
    if len(train_dataset) > 0:
        print("\ntrain sample")
        describe_sample(train_dataset[0])
    if len(val_dataset) > 0:
        print("\nval sample")
        describe_sample(val_dataset[0])


if __name__ == "__main__":
    main()
