"""
Utility functions for audio encoding pipeline.

This module provides helper functions for:
- Padding and masking operations
- Embedding alignment and matching
- Loss computations
- Device management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class PaddingUtils:
    """Utilities for batch padding and mask generation."""

    @staticmethod
    def pad_batch_audio(
        audio_list: list[torch.Tensor], target_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a list of audio tensors to same length.

        Args:
            audio_list (list[torch.Tensor]): List of audio tensors.
                Each tensor shape: (num_samples,)
            target_length (int, optional): Target length. If None, uses max length.

        Returns:
            padded_audio (torch.Tensor): Padded audio batch.
                Shape: (batch_size, target_length)
            mask (torch.Tensor): Boolean attention mask (True for real, False for padding).
                Shape: (batch_size, target_length)
        """
        if target_length is None:
            target_length = max(audio.shape[0] for audio in audio_list)

        batch_size = len(audio_list)
        padded = torch.zeros(batch_size, target_length)
        mask = torch.ones(batch_size, target_length, dtype=torch.bool)

        for i, audio in enumerate(audio_list):
            seq_len = min(audio.shape[0], target_length)
            padded[i, :seq_len] = audio[:seq_len]
            mask[i, seq_len:] = False

        return padded, mask

    @staticmethod
    def create_padding_mask(
        lengths: torch.Tensor, max_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create attention mask from sequence lengths.

        Args:
            lengths (torch.Tensor): Actual sequence lengths.
                Shape: (batch_size,)
            max_len (int, optional): Maximum sequence length.

        Returns:
            mask (torch.Tensor): Boolean mask (True for valid, False for padding).
                Shape: (batch_size, max_len)
        """
        if max_len is None:
            max_len = lengths.max().item()

        batch_size = lengths.shape[0]
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)

        return mask

    @staticmethod
    def apply_mask_to_embeddings(
        embeddings: torch.Tensor, mask: torch.Tensor, mask_value: float = 0.0
    ) -> torch.Tensor:
        """
        Apply mask to embeddings (zero out padding positions).

        Args:
            embeddings (torch.Tensor): Embeddings. Shape: (batch, seq_len, dim)
            mask (torch.Tensor): Boolean mask. Shape: (batch, seq_len)
            mask_value (float): Value to use for masked positions.

        Returns:
            masked_embeddings (torch.Tensor): Masked embeddings.
        """
        # Expand mask to match embedding dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)
        masked = embeddings * mask_expanded + mask_value * (~mask_expanded)
        return masked


class EmbeddingAlignment:
    """Utilities for aligning embeddings across modalities."""

    @staticmethod
    def interpolate_embeddings(
        embeddings: torch.Tensor,
        source_fps: float,
        target_fps: float,
        mode: str = "linear",
    ) -> torch.Tensor:
        """
        Interpolate embeddings to match target frame rate.

        Useful for aligning audio embeddings (HuBERT extraction rate) with
        video frame rate.

        Args:
            embeddings (torch.Tensor): Input embeddings.
                Shape: (batch, source_frames, dim)
            source_fps (float): Source frame rate (frames per second).
            target_fps (float): Target frame rate.
            mode (str): Interpolation mode ("linear" or "nearest").

        Returns:
            interpolated (torch.Tensor): Interpolated embeddings.
                Shape: (batch, target_frames, dim)
        """
        batch_size, source_frames, dim = embeddings.shape

        # Calculate target number of frames
        target_frames = int(source_frames * target_fps / source_fps)

        # Reshape for interpolation: (batch*dim, 1, source_frames)
        embeddings_reshaped = embeddings.permute(0, 2, 1).reshape(
            batch_size * dim, 1, source_frames
        )

        # Interpolate using grid_sample or F.interpolate
        interpolated = F.interpolate(
            embeddings_reshaped,
            size=target_frames,
            mode=mode,
            align_corners=False if mode == "linear" else None,
        )

        # Reshape back
        interpolated = interpolated.reshape(batch_size, dim, target_frames).permute(
            0, 2, 1
        )

        return interpolated

    @staticmethod
    def match_sequence_lengths(
        audio_embeddings: torch.Tensor,
        video_embeddings: torch.Tensor,
        align_to: str = "video",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align audio and video embeddings to same sequence length.

        Args:
            audio_embeddings (torch.Tensor): Shape: (batch, audio_frames, audio_dim)
            video_embeddings (torch.Tensor): Shape: (batch, video_frames, video_dim)
            align_to (str): "video" to align to video frames, "audio" to align to audio.

        Returns:
            audio_aligned (torch.Tensor): Aligned audio embeddings.
            video_aligned (torch.Tensor): Aligned video embeddings.
        """
        audio_frames = audio_embeddings.shape[1]
        video_frames = video_embeddings.shape[1]

        if align_to == "video":
            # Interpolate audio to match video frame count
            audio_aligned = F.interpolate(
                audio_embeddings.permute(0, 2, 1),
                size=video_frames,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
            video_aligned = video_embeddings
        elif align_to == "audio":
            # Interpolate video to match audio frame count
            video_aligned = F.interpolate(
                video_embeddings.permute(0, 2, 1),
                size=audio_frames,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
            audio_aligned = audio_embeddings
        else:
            raise ValueError("align_to must be 'video' or 'audio'")

        return audio_aligned, video_aligned


class LossFunctions:
    """Custom loss functions for audio-visual tasks."""

    @staticmethod
    def frame_level_bce_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute frame-level binary cross-entropy loss.

        Args:
            predictions (torch.Tensor): Model predictions. Shape: (batch, frames)
            targets (torch.Tensor): Ground truth labels. Shape: (batch, frames)
            mask (torch.Tensor, optional): Frame mask. Shape: (batch, frames)
            pos_weight (float): Weight for positive class (for class imbalance).

        Returns:
            loss (torch.Tensor): Scalar loss value.
        """
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight))
        loss = loss_fn(predictions, targets.float())

        if mask is not None:
            loss = loss * mask.float()
            loss = loss.sum() / (mask.float().sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    @staticmethod
    def temporal_smoothness_loss(
        embeddings: torch.Tensor, lambda_smooth: float = 0.01
    ) -> torch.Tensor:
        """
        Regularization loss to encourage temporal smoothness in embeddings.

        Penalizes sharp changes between consecutive frames.

        Args:
            embeddings (torch.Tensor): Embeddings. Shape: (batch, frames, dim)
            lambda_smooth (float): Weight for smoothness loss.

        Returns:
            loss (torch.Tensor): Scalar smoothness loss.
        """
        # Compute difference between consecutive frames
        diff = embeddings[:, 1:, :] - embeddings[:, :-1, :]
        # L2 norm along feature dimension
        diff_norm = torch.norm(diff, dim=2)  # (batch, frames-1)
        # Mean over batch and frames
        smoothness_loss = diff_norm.mean()
        return lambda_smooth * smoothness_loss

    @staticmethod
    def contrastive_loss(
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Contrastive loss for matching embeddings across modalities.

        Args:
            embeddings1 (torch.Tensor): First set of embeddings. Shape: (batch, dim)
            embeddings2 (torch.Tensor): Second set of embeddings. Shape: (batch, dim)
            labels (torch.Tensor): Binary labels (same/different). Shape: (batch,)
            temperature (float): Temperature for softmax scaling.

        Returns:
            loss (torch.Tensor): Scalar contrastive loss.
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        # Cosine similarity
        similarity = torch.matmul(embeddings1, embeddings2.t()) / temperature

        # Contrastive loss
        loss_fn = nn.CrossEntropyLoss()
        targets = torch.arange(len(embeddings1))
        loss = loss_fn(similarity, targets)

        return loss


class DeviceUtils:
    """Utilities for device management."""

    @staticmethod
    def get_device(prefer_cuda: bool = True) -> torch.device:
        """
        Get appropriate device (GPU or CPU).

        Args:
            prefer_cuda (bool): Prefer CUDA if available.

        Returns:
            device (torch.device): Selected device.
        """
        if prefer_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU device")

        return device

    @staticmethod
    def move_batch_to_device(batch: dict, device: torch.device) -> dict:
        """
        Move all tensors in a batch to specified device.

        Args:
            batch (dict): Batch dictionary.
            device (torch.device): Target device.

        Returns:
            batch_moved (dict): Batch on target device.
        """
        batch_moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_moved[key] = value.to(device)
            else:
                batch_moved[key] = value
        return batch_moved


class MetricsUtils:
    """Utilities for computing metrics."""

    @staticmethod
    def compute_frame_accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute frame-level accuracy.

        Args:
            predictions (torch.Tensor): Model predictions (logits). Shape: (batch, frames)
            targets (torch.Tensor): Ground truth labels. Shape: (batch, frames)
            threshold (float): Classification threshold.

        Returns:
            accuracy (float): Frame accuracy in range [0, 1].
        """
        pred_binary = (predictions > threshold).float()
        correct = (pred_binary == targets.float()).float()
        accuracy = correct.mean().item()
        return accuracy

    @staticmethod
    def compute_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict:
        """
        Compute multiple classification metrics.

        Args:
            predictions (torch.Tensor): Predictions. Shape: (batch*frames,)
            targets (torch.Tensor): Targets. Shape: (batch*frames,)
            threshold (float): Classification threshold.

        Returns:
            metrics (dict): Dictionary with TP, FP, FN, TN, precision, recall, F1.
        """
        pred_binary = (predictions > threshold).int()
        targets = targets.int()

        tp = ((pred_binary == 1) & (targets == 1)).sum().item()
        fp = ((pred_binary == 1) & (targets == 0)).sum().item()
        fn = ((pred_binary == 0) & (targets == 1)).sum().item()
        tn = ((pred_binary == 0) & (targets == 0)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }


# Example usage
if __name__ == "__main__":
    # Test padding
    audio_list = [torch.randn(16000), torch.randn(12000), torch.randn(20000)]
    padded, mask = PaddingUtils.pad_batch_audio(audio_list)
    print(f"Padded shape: {padded.shape}, Mask shape: {mask.shape}")

    # Test embedding alignment
    audio_emb = torch.randn(2, 400, 768)  # audio frames
    video_emb = torch.randn(2, 300, 512)  # video frames
    audio_aligned, video_aligned = EmbeddingAlignment.match_sequence_lengths(
        audio_emb, video_emb, align_to="video"
    )
    print(f"Aligned audio shape: {audio_aligned.shape}")
    print(f"Aligned video shape: {video_aligned.shape}")

    # Test device
    device = DeviceUtils.get_device(prefer_cuda=False)
    print(f"Device: {device}")
