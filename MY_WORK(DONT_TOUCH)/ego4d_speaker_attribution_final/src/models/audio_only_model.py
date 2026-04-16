import torch
import torch.nn as nn

from .benchmark_blocks import AudioTokenEncoder, SequenceClassifierHead, masked_mean


class AudioOnlyTTMModel(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.audio_encoder = AudioTokenEncoder(hidden_dim=hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = SequenceClassifierHead(hidden_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, audio: torch.Tensor, audio_mask: torch.Tensor, **_) -> torch.Tensor:
        audio_tokens = self.audio_encoder(audio)
        encoded = self.temporal(audio_tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)
