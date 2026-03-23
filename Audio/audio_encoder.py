"""
Audio Encoder Module for Ego4D TTM Task.

This module implements a HuBERT-based audio encoder that extracts frame-level
audio embeddings suitable for cross-modal fusion with visual features.

Why HuBERT?
-----------
HuBERT (Hidden-Unit BERT) is a self-supervised speech representation model
that captures rich phonetic and speaker information without requiring labels.
Pre-trained on large-scale unlabeled speech data (960h of LibriSpeech),
it provides robust embeddings for downstream tasks like TTM classification.

Key advantages:
- Strong speaker/audio understanding without fine-tuning
- Low-level feature extraction (better for detecting speech presence)
- Temporal consistency (good for video frame alignment)
- Minimal computational overhead compared to ASR models
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from typing import Optional, Tuple
import warnings

from config import AudioConfig


class AudioEncoder(nn.Module):
    """
    HuBERT-based audio encoder for frame-level audio feature extraction.

    This encoder loads a pretrained HuBERT model and extracts embeddings
    from the last hidden layer. Supports optional pooling operations and
    variable-length sequences.

    Attributes:
        model_name (str): HuggingFace model identifier.
        embedding_dim (int): Output embedding dimension.
        freeze_encoder (bool): Whether to freeze encoder weights.
        use_attention_pooling (bool): Use attention pooling instead of mean.
    """

    def __init__(
        self,
        config: AudioConfig,
        return_all_hidden_states: bool = False,
    ):
        """
        Initialize the Audio Encoder.

        Args:
            config (AudioConfig): Configuration object with hyperparameters.
            return_all_hidden_states (bool): If True, return all hidden states
                from the model. Default: False (returns last hidden state).

        Raises:
            ValueError: If model cannot be loaded from HuggingFace.
        """
        super().__init__()

        self.config = config
        self.return_all_hidden_states = return_all_hidden_states

        # Load pretrained HuBERT model
        try:
            print(f"Loading pretrained model: {config.model_name}")
            self.model = AutoModel.from_pretrained(config.model_name)
            self.processor = AutoProcessor.from_pretrained(config.model_name)
        except Exception as e:
            raise ValueError(
                f"Failed to load model {config.model_name}. "
                f"Ensure it's available on HuggingFace Hub. Error: {e}"
            )

        # Model properties
        self.embedding_dim = config.embedding_dim
        self.freeze_encoder = config.freeze_encoder

        # Freeze encoder if specified
        if self.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Audio encoder weights frozen (not trainable).")
        else:
            print("Audio encoder weights are trainable.")

        # Optional attention pooling layer
        if config.use_attention_pooling:
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=self.embedding_dim,
                num_heads=8,
                batch_first=True,
                dropout=0.1,
            )
            self.attention_weights_store = None
        else:
            self.attention_pool = None

    def forward(
        self,
        audio_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the audio encoder.

        **Important: Output shape alignment**
        - Input shape: (batch_size, seq_length) -> raw audio waveform
        - Output shape: (batch_size, num_frames, embedding_dim)

        This ensures alignment with video frames for cross-modal fusion.

        Args:
            audio_values (torch.Tensor): Raw audio waveform.
                Shape: (batch_size, num_samples)
            attention_mask (torch.Tensor, optional): Attention mask for
                padding tokens. Shape: (batch_size, num_samples)

        Returns:
            embeddings (torch.Tensor): Audio frame embeddings.
                Shape: (batch_size, num_frames, embedding_dim)
            attention_weights (torch.Tensor or None): Attention weights if
                attention pooling is used. Shape: (batch_size, num_frames).
        """

        # Forward pass through HuBERT
        outputs = self.model(
            audio_values,
            attention_mask=attention_mask,
            output_hidden_states=self.return_all_hidden_states,
            return_dict=True,
        )

        # Extract embeddings from last hidden state
        # Shape: (batch_size, num_frames, embedding_dim)
        embeddings = outputs.last_hidden_state

        attention_weights = None

        # Apply attention pooling if configured
        if self.attention_pool is not None:
            embeddings, attention_weights = self.attention_pool(
                embeddings, embeddings, embeddings, need_weights=True
            )
            # attention_weights shape: (batch_size * num_heads, num_frames, num_frames)
            # Aggregate across heads
            batch_size, seq_len, _ = embeddings.shape
            attention_weights = attention_weights.view(
                batch_size, -1, seq_len, seq_len
            ).mean(dim=1)  # (batch_size, num_frames, num_frames)
            # Take mean across source dimension
            attention_weights = attention_weights.mean(dim=2)  # (batch_size, num_frames)

        return embeddings, attention_weights

    def extract_embeddings(
        self,
        audio_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "none",
    ) -> torch.Tensor:
        """
        Extract embeddings with optional pooling.

        Args:
            audio_values (torch.Tensor): Raw audio waveform.
                Shape: (batch_size, num_samples)
            attention_mask (torch.Tensor, optional): Padding mask.
            pooling (str): Pooling strategy.
                - "none": Return full sequence (num_frames, embedding_dim)
                - "mean": Return mean-pooled embedding (embedding_dim,)
                - "cls": Return first token embedding

        Returns:
            embeddings (torch.Tensor): Pooled embeddings.
        """
        embeddings, _ = self.forward(audio_values, attention_mask)

        if pooling == "none":
            return embeddings  # (batch_size, num_frames, embedding_dim)
        elif pooling == "mean":
            if attention_mask is not None:
                # Properly handle masked mean pooling
                mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
                return (embeddings * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                return embeddings.mean(dim=1)  # (batch_size, embedding_dim)
        elif pooling == "cls":
            return embeddings[:, 0, :]  # (batch_size, embedding_dim)
        else:
            raise ValueError(
                f"Unknown pooling strategy: {pooling}. "
                f"Choose from ['none', 'mean', 'cls']"
            )

    def get_embedding_dim(self) -> int:
        """Get output embedding dimension."""
        return self.embedding_dim

    def enable_train_mode(self):
        """Enable training mode (gradients active)."""
        self.train()
        if self.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

    def enable_eval_mode(self):
        """Enable evaluation mode."""
        self.eval()


# Example usage
if __name__ == "__main__":
    from config import AudioConfig

    # Initialize config
    config = AudioConfig()

    # Create encoder
    encoder = AudioEncoder(config)

    # Create dummy audio input (batch_size=2, num_samples=16000)
    dummy_audio = torch.randn(2, 16000)

    # Forward pass
    embeddings, attention_weights = encoder(dummy_audio)

    print(f"Input shape: {dummy_audio.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    if attention_weights is not None:
        print(f"Attention weights shape: {attention_weights.shape}")

    # Extract with mean pooling
    pooled = encoder.extract_embeddings(dummy_audio, pooling="mean")
    print(f"Mean-pooled shape: {pooled.shape}")
