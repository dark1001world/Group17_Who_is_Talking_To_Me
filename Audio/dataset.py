"""
Dataset module for Ego4D TTM task.

This module provides PyTorch Dataset classes for loading audio-visual data
with frame-level labels for the Talking To Me classification task.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np

from preprocessing import AudioProcessor, FrameAlignmentUtils
from config import AudioConfig


class Ego4D_TTM_Dataset(Dataset):
    """
    PyTorch Dataset for Ego4D Talking To Me (TTM) task.

    This dataset loads:
    - Audio files (.wav)
    - Video frame count / duration
    - Frame-level labels (binary: speaking to camera or not)

    Expected annotation format (JSON):
    {
        "video_id": {
            "audio_path": "path/to/audio.wav",
            "duration_frames": 300,
            "segments": [
                {"start_frame": 10, "end_frame": 50, "label": 1},
                {"start_frame": 100, "end_frame": 150, "label": 0},
            ]
        }
    }
    """

    def __init__(
        self,
        annotation_file: str,
        audio_root: str,
        config: AudioConfig,
        video_fps: float = 30.0,
        split: str = "train",
        train_split: float = 0.8,
        val_split: float = 0.1,
    ):
        """
        Initialize Ego4D TTM Dataset.

        Args:
            annotation_file (str): Path to JSON annotation file.
            audio_root (str): Root directory containing audio files.
            config (AudioConfig): Audio configuration.
            video_fps (float): Video frame rate (frames per second).
            split (str): Dataset split ("train", "val", or "test").
            train_split (float): Fraction of data for training.
            val_split (float): Fraction of data for validation.
        """
        self.annotation_file = Path(annotation_file)
        self.audio_root = Path(audio_root)
        self.config = config
        self.video_fps = video_fps
        self.split = split

        # Load annotations
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

        # Create processor
        self.processor = AudioProcessor(config)
        self.aligner = FrameAlignmentUtils(video_fps=video_fps)

        # Split dataset
        self.video_ids = list(self.annotations.keys())
        num_videos = len(self.video_ids)
        train_end = int(num_videos * train_split)
        val_end = int(num_videos * (train_split + val_split))

        if split == "train":
            self.video_ids = self.video_ids[:train_end]
        elif split == "val":
            self.video_ids = self.video_ids[train_end:val_end]
        elif split == "test":
            self.video_ids = self.video_ids[val_end:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        print(f"Loaded {len(self.video_ids)} videos for {split} split")

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Sample index.

        Returns:
            sample (dict): Dictionary with keys:
                - "video_id": str
                - "audio": torch.Tensor, shape (1, num_samples)
                - "audio_mask": torch.Tensor, shape (1, num_samples)
                - "labels": torch.Tensor, shape (num_frames,)
                - "num_frames": int
        """
        video_id = self.video_ids[idx]
        annotation = self.annotations[video_id]

        # Load audio
        audio_path = self.audio_root / annotation["audio_path"]
        try:
            audio, audio_mask = self.processor.preprocess_audio(
                str(audio_path), self.config.max_audio_length * self.config.sample_rate
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load audio for {video_id}: {e}")

        # Squeeze channel dimension if present
        if audio.shape[0] == 1:
            audio = audio.squeeze(0)  # (num_samples,)
            audio_mask = audio_mask.squeeze(0)

        # Get number of video frames
        num_frames = annotation["duration_frames"]

        # Create frame-level labels
        labels = self._create_frame_labels(annotation["segments"], num_frames)

        sample = {
            "video_id": video_id,
            "audio": audio,
            "audio_mask": audio_mask,
            "labels": labels,
            "num_frames": num_frames,
        }

        return sample

    def _create_frame_labels(
        self, segments: List[Dict], num_frames: int
    ) -> torch.Tensor:
        """
        Create frame-level binary labels from segment annotations.

        Args:
            segments (list): List of segment dicts with keys:
                - "start_frame": int
                - "end_frame": int
                - "label": int (0 or 1)
            num_frames (int): Total number of frames.

        Returns:
            labels (torch.Tensor): Binary labels per frame. Shape: (num_frames,)
        """
        labels = torch.zeros(num_frames, dtype=torch.long)

        for segment in segments:
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            label = int(segment["label"])

            # Clamp to valid range
            start = max(0, min(start, num_frames - 1))
            end = max(0, min(end, num_frames))

            labels[start:end] = label

        return labels


class AudioOnlyDataset(Dataset):
    """
    Simplified Dataset for audio-only experiments.

    Loads audio files with binary labels (TTM or not).
    """

    def __init__(
        self,
        audio_dir: str,
        label_file: str,
        config: AudioConfig,
    ):
        """
        Initialize audio-only dataset.

        Expected label file format (JSON):
        {
            "audio_file1.wav": 1,
            "audio_file2.wav": 0,
            ...
        }

        Args:
            audio_dir (str): Directory containing audio files.
            label_file (str): Path to JSON file with audio labels.
            config (AudioConfig): Audio configuration.
        """
        self.audio_dir = Path(audio_dir)
        self.config = config

        # Load labels
        with open(label_file, "r") as f:
            self.labels = json.load(f)

        self.audio_files = list(self.labels.keys())
        self.processor = AudioProcessor(config)

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get audio and label.

        Returns:
            sample (dict): Keys: "audio", "audio_mask", "label"
        """
        audio_file = self.audio_files[idx]
        audio_path = self.audio_dir / audio_file

        # Load and preprocess audio
        audio, audio_mask = self.processor.preprocess_audio(str(audio_path))
        audio = audio.squeeze(0)
        audio_mask = audio_mask.squeeze(0)

        # Get label
        label = self.labels[audio_file]

        sample = {
            "audio_file": audio_file,
            "audio": audio,
            "audio_mask": audio_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }

        return sample


def collate_fn_ego4d(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for Ego4D TTM dataset.

    Handles variable-length audio sequences by padding.

    Args:
        batch (list): List of samples from dataset.

    Returns:
        collated_batch (dict): Padded batch with keys:
            - "video_ids": list
            - "audio": torch.Tensor, shape (batch_size, max_samples)
            - "audio_mask": torch.Tensor, shape (batch_size, max_samples)
            - "labels": list of torch.Tensor (variable length)
            - "num_frames": list
    """
    video_ids = [sample["video_id"] for sample in batch]
    audio_list = [sample["audio"] for sample in batch]
    mask_list = [sample["audio_mask"] for sample in batch]
    labels_list = [sample["labels"] for sample in batch]
    num_frames_list = [sample["num_frames"] for sample in batch]

    # Pad audio sequences
    max_len = max(audio.shape[0] for audio in audio_list)
    batch_size = len(batch)

    audio_padded = torch.zeros(batch_size, max_len)
    mask_padded = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, audio in enumerate(audio_list):
        seq_len = audio.shape[0]
        audio_padded[i, :seq_len] = audio
        mask_padded[i, seq_len:] = False

    collated = {
        "video_ids": video_ids,
        "audio": audio_padded,
        "audio_mask": mask_padded,
        "labels": labels_list,  # Keep as list (variable length per video)
        "num_frames": num_frames_list,
    }

    return collated


def collate_fn_audio_only(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for audio-only dataset.

    Args:
        batch (list): List of samples.

    Returns:
        collated_batch (dict): Padded batch.
    """
    audio_files = [sample["audio_file"] for sample in batch]
    audio_list = [sample["audio"] for sample in batch]
    mask_list = [sample["audio_mask"] for sample in batch]
    labels = torch.stack([sample["label"] for sample in batch])

    # Pad audio
    max_len = max(audio.shape[0] for audio in audio_list)
    batch_size = len(batch)

    audio_padded = torch.zeros(batch_size, max_len)
    mask_padded = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, audio in enumerate(audio_list):
        seq_len = audio.shape[0]
        audio_padded[i, :seq_len] = audio
        mask_padded[i, seq_len:] = False

    collated = {
        "audio_files": audio_files,
        "audio": audio_padded,
        "audio_mask": mask_padded,
        "labels": labels,
    }

    return collated


def create_dataloaders(
    annotation_file: str,
    audio_root: str,
    config: AudioConfig,
    train_batch_size: int = 32,
    val_batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders.

    Args:
        annotation_file (str): Path to annotation JSON.
        audio_root (str): Root directory for audio files.
        config (AudioConfig): Audio configuration.
        train_batch_size (int): Training batch size.
        val_batch_size (int): Validation/test batch size.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        train_loader, val_loader, test_loader (DataLoader)
    """
    train_dataset = Ego4D_TTM_Dataset(
        annotation_file, audio_root, config, split="train"
    )
    val_dataset = Ego4D_TTM_Dataset(
        annotation_file, audio_root, config, split="val"
    )
    test_dataset = Ego4D_TTM_Dataset(
        annotation_file, audio_root, config, split="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_ego4d,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_ego4d,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_ego4d,
    )

    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    from config import AudioConfig

    config = AudioConfig()

    # Example: Create dummy annotation file
    dummy_annotations = {
        "video_001": {
            "audio_path": "audio_001.wav",
            "duration_frames": 300,
            "segments": [
                {"start_frame": 10, "end_frame": 50, "label": 1},
                {"start_frame": 100, "end_frame": 150, "label": 1},
            ],
        },
    }

    with open("dummy_annotations.json", "w") as f:
        json.dump(dummy_annotations, f)

    print("Dataset module loaded successfully")
    print(
        "Use create_dataloaders() to create train/val/test dataloaders for your data"
    )
