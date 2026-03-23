"""
Audio Preprocessing Module.

This module handles audio loading, resampling, padding, and frame-level
alignment utilities for the Ego4D TTM task.
"""

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

from config import AudioConfig


class AudioProcessor:
    """
    Handles audio loading, resampling, and preprocessing.

    This class provides utilities for:
    - Loading audio files from disk
    - Resampling to target sample rate (16kHz for HuBERT)
    - Padding/truncating to fixed length
    - Creating frame-aligned segments
    """

    def __init__(self, config: AudioConfig):
        """
        Initialize the audio processor.

        Args:
            config (AudioConfig): Configuration with audio settings.
        """
        self.config = config
        self.sample_rate = config.sample_rate
        self.window_samples = config.get_window_samples()
        self.stride_samples = config.get_stride_samples()
        self.max_samples = int(config.max_audio_length * config.sample_rate)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file from disk.

        Supports: .wav, .mp3, .flac, and other formats supported by torchaudio.

        Args:
            audio_path (str): Path to audio file.

        Returns:
            audio (torch.Tensor): Audio waveform. Shape: (channels, num_samples)
            sample_rate (int): Original sample rate.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If audio file cannot be decoded.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            audio, sample_rate = torchaudio.load(str(audio_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio from {audio_path}. Error: {e}"
            )

        return audio, sample_rate

    def resample_to_target(
        self, audio: torch.Tensor, orig_sample_rate: int
    ) -> torch.Tensor:
        """
        Resample audio to target sample rate (16kHz for HuBERT).

        Args:
            audio (torch.Tensor): Audio waveform. Shape: (channels, num_samples)
            orig_sample_rate (int): Original sample rate.

        Returns:
            audio_resampled (torch.Tensor): Resampled audio.
                Shape: (channels, new_num_samples)
        """
        if orig_sample_rate == self.sample_rate:
            return audio

        # Create resampler
        resampler = T.Resample(orig_sample_rate, self.sample_rate)
        audio_resampled = resampler(audio)

        return audio_resampled

    def to_mono(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert stereo audio to mono by averaging channels.

        Args:
            audio (torch.Tensor): Audio waveform. Shape: (channels, num_samples)

        Returns:
            audio_mono (torch.Tensor): Mono audio. Shape: (1, num_samples)
        """
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        return audio

    def pad_or_truncate(
        self, audio: torch.Tensor, target_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad or truncate audio to target length.

        Args:
            audio (torch.Tensor): Audio waveform. Shape: (1, num_samples)
            target_length (int): Target length in samples.

        Returns:
            audio_processed (torch.Tensor): Padded/truncated audio.
                Shape: (1, target_length)
            mask (torch.Tensor): Attention mask (1 for real samples, 0 for padding).
                Shape: (1, target_length)
        """
        current_length = audio.shape[-1]

        if current_length >= target_length:
            # Truncate
            audio = audio[..., :target_length]
            mask = torch.ones_like(audio)
        else:
            # Pad with zeros
            pad_amount = target_length - current_length
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
            # Create mask: 1 for real samples, 0 for padding
            mask = torch.cat(
                [
                    torch.ones(1, current_length),
                    torch.zeros(1, pad_amount),
                ],
                dim=1,
            )

        return audio, mask

    def preprocess_audio(
        self, audio_path: str, target_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete preprocessing pipeline.

        Steps:
        1. Load audio file
        2. Resample to 16kHz
        3. Convert to mono
        4. Pad/truncate to target length

        Args:
            audio_path (str): Path to audio file.
            target_length (int, optional): Target length in samples.
                If None, uses max_samples from config.

        Returns:
            audio (torch.Tensor): Preprocessed audio. Shape: (1, target_length)
            mask (torch.Tensor): Attention mask. Shape: (1, target_length)
        """
        if target_length is None:
            target_length = self.max_samples

        # Load audio
        audio, orig_sr = self.load_audio(audio_path)

        # Resample
        audio = self.resample_to_target(audio, orig_sr)

        # Convert to mono
        audio = self.to_mono(audio)

        # Pad/truncate
        audio, mask = self.pad_or_truncate(audio, target_length)

        return audio, mask

    def get_num_frames(self, num_samples: int) -> int:
        """
        Calculate number of frames (HuBERT output frames) for given samples.

        HuBERT applies a convolutional feature extractor that reduces temporal
        dimension. By default, it reduces by a factor of 4 (stride=2, 2 layers).

        Args:
            num_samples (int): Number of audio samples at 16kHz.

        Returns:
            num_frames (int): Number of feature frames output by HuBERT.
        """
        # HuBERT uses Conv1d with kernel=10, stride=5 for first layer
        # Then additional stride-2 conv layers
        # Effective reduction: 4x (standard HuBERT-base)
        hubert_reduction_factor = 4
        return num_samples // hubert_reduction_factor

    def get_frame_timestamps(self, num_frames: int) -> np.ndarray:
        """
        Get timestamps (in seconds) for each frame.

        Useful for aligning with video frames.

        Args:
            num_frames (int): Number of HuBERT output frames.

        Returns:
            timestamps (np.ndarray): Timestamps in seconds. Shape: (num_frames,)
        """
        # Each frame represents ~20ms of audio (16000 samples / 4 reduction factor / 50fps equivalent)
        frame_duration = 4 / self.sample_rate  # 4 samples per frame at 16kHz
        timestamps = np.arange(num_frames) * frame_duration
        return timestamps


class FrameAlignmentUtils:
    """
    Utilities for aligning audio frames with video frames.

    For Ego4D, videos are typically at 30fps, so each frame spans ~33ms.
    This module helps align audio embeddings to video frame timing.
    """

    def __init__(self, video_fps: float = 30.0, audio_sample_rate: int = 16000):
        """
        Initialize frame alignment utilities.

        Args:
            video_fps (float): Video frame rate (frames per second).
            audio_sample_rate (int): Audio sample rate (Hz).
        """
        self.video_fps = video_fps
        self.audio_sample_rate = audio_sample_rate
        self.frame_duration_ms = 1000 / video_fps  # ms per video frame

    def audio_samples_to_frame_idx(self, num_samples: int) -> int:
        """
        Convert audio sample count to corresponding video frame index.

        Args:
            num_samples (int): Number of audio samples at 16kHz.

        Returns:
            frame_idx (int): Corresponding video frame index.
        """
        time_seconds = num_samples / self.audio_sample_rate
        frame_idx = int(time_seconds * self.video_fps)
        return frame_idx

    def frame_idx_to_audio_samples(self, frame_idx: int) -> int:
        """
        Convert video frame index to audio sample position.

        Args:
            frame_idx (int): Video frame index.

        Returns:
            num_samples (int): Corresponding audio sample position.
        """
        time_seconds = frame_idx / self.video_fps
        num_samples = int(time_seconds * self.audio_sample_rate)
        return num_samples

    def create_frame_level_targets(
        self,
        segment_start_ms: float,
        segment_end_ms: float,
        num_video_frames: int,
        label: int,
    ) -> torch.Tensor:
        """
        Create frame-level binary labels for a segment.

        Useful for converting segment-level annotations to frame-level labels.

        Args:
            segment_start_ms (float): Start time of segment (ms).
            segment_end_ms (float): End time of segment (ms).
            num_video_frames (int): Total number of video frames.
            label (int): Binary label (0 or 1).

        Returns:
            frame_labels (torch.Tensor): Frame-level labels.
                Shape: (num_video_frames,)
        """
        frame_labels = torch.zeros(num_video_frames, dtype=torch.long)

        # Convert ms to frame indices
        start_frame = int((segment_start_ms / 1000) * self.video_fps)
        end_frame = int((segment_end_ms / 1000) * self.video_fps)

        # Clamp to valid range
        start_frame = max(0, min(start_frame, num_video_frames - 1))
        end_frame = max(0, min(end_frame, num_video_frames))

        # Set labels for frames in segment
        frame_labels[start_frame:end_frame] = label

        return frame_labels


# Example usage
if __name__ == "__main__":
    config = AudioConfig()
    processor = AudioProcessor(config)

    # Example: Create dummy audio and preprocess
    dummy_audio_path = "sample_audio.wav"
    # (In real use, replace with actual audio file path)

    # For testing, create a dummy audio file
    sample_rate = 16000
    duration = 2  # seconds
    dummy_audio = torch.randn(1, sample_rate * duration)
    torchaudio.save(dummy_audio_path, dummy_audio, sample_rate)

    # Preprocess
    audio, mask = processor.preprocess_audio(dummy_audio_path)
    print(f"Preprocessed audio shape: {audio.shape}")
    print(f"Attention mask shape: {mask.shape}")

    # Frame alignment
    aligner = FrameAlignmentUtils(video_fps=30.0)
    num_frames = processor.get_num_frames(audio.shape[-1])
    print(f"Number of HuBERT frames: {num_frames}")

    # Clean up
    import os

    os.remove(dummy_audio_path)
