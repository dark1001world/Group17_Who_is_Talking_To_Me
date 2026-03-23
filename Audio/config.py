"""
Configuration module for Audio Encoder Pipeline.

This module centralizes all hyperparameters and configuration settings
for the audio encoding pipeline used in the Ego4D TTM (Talking To Me) task.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioConfig:
    """
    Configuration for audio preprocessing and encoding.

    Attributes:
        sample_rate (int): Target sample rate in Hz. HuBERT expects 16kHz.
        window_size (float): Duration of each audio window in seconds.
        stride (float): Stride between consecutive windows in seconds.
        n_fft (int): FFT window size for spectrogram (if needed).
        n_mels (int): Number of mel frequency bins.
        model_name (str): HuggingFace model name for pretrained audio encoder.
        embedding_dim (int): Dimension of extracted embeddings.
        freeze_encoder (bool): Whether to freeze the encoder weights.
        use_attention_pooling (bool): Use attention pooling instead of mean pooling.
    """

    # Audio preprocessing
    sample_rate: int = 16000  # HuBERT requires 16kHz
    window_size: float = 0.04  # 40ms window (matches video frame rate ~25fps)
    stride: float = 0.04  # 40ms stride for frame-level alignment
    n_fft: int = 400  # FFT for spectrogram
    n_mels: int = 80  # Mel bands

    # Model configuration
    model_name: str = "facebook/hubert-base-ls960"  # Primary: HuBERT
    # Alternative: "facebook/wav2vec2-base"
    embedding_dim: int = 768  # HuBERT-base output dimension

    # Training configuration
    freeze_encoder: bool = False  # Set to True if fine-tuning downstream classifier
    use_attention_pooling: bool = False  # Use attention-based pooling

    # Dataset
    max_audio_length: float = 60.0  # Maximum audio length in seconds
    padding_type: str = "constant"  # "constant" or "center"

    def get_window_samples(self) -> int:
        """Calculate number of audio samples in one window."""
        return int(self.sample_rate * self.window_size)

    def get_stride_samples(self) -> int:
        """Calculate number of audio samples in one stride."""
        return int(self.sample_rate * self.stride)


@dataclass
class TrainingConfig:
    """
    Configuration for training the audio encoder.

    Attributes:
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for AdamW optimizer.
        warmup_steps (int): Number of warmup steps for scheduler.
        max_grad_norm (float): Maximum gradient norm for clipping.
        device (str): Device to train on ('cuda' or 'cpu').
    """

    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    device: str = "cuda"  # Auto-select in practice

    # Checkpointing
    save_every_n_steps: int = 500
    validate_every_n_steps: int = 100


@dataclass
class DataConfig:
    """
    Configuration for data loading.

    Attributes:
        data_root (str): Root directory for audio files.
        annotation_file (str): Path to annotation JSON file.
        train_split (float): Fraction of data for training.
        val_split (float): Fraction of data for validation.
        num_workers (int): Number of workers for DataLoader.
    """

    data_root: str = "./data/audio"
    annotation_file: str = "./data/annotations.json"
    train_split: float = 0.8
    val_split: float = 0.1
    # test_split = 0.1 (remainder)
    num_workers: int = 4


def get_config() -> tuple[AudioConfig, TrainingConfig, DataConfig]:
    """
    Get all configuration objects.

    Returns:
        Tuple of (AudioConfig, TrainingConfig, DataConfig)
    """
    return AudioConfig(), TrainingConfig(), DataConfig()


# Example: Override default configs
if __name__ == "__main__":
    audio_cfg, train_cfg, data_cfg = get_config()
    print(f"Audio Config: {audio_cfg}")
    print(f"Training Config: {train_cfg}")
    print(f"Data Config: {data_cfg}")
