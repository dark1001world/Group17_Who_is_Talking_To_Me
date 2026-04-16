import torch
import torch.nn as nn

from .benchmark_blocks import SequenceClassifierHead, VisualTokenEncoder, masked_mean


class VisualOnlyTTMModel(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.visual_encoder = VisualTokenEncoder(hidden_dim=hidden_dim, dropout=dropout)
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

    def forward(self, video: torch.Tensor, face_mask: torch.Tensor, frame_mask: torch.Tensor, **_) -> torch.Tensor:
        visual_tokens = self.visual_encoder(video)
        valid_mask = face_mask & frame_mask
        encoded = self.temporal(visual_tokens, src_key_padding_mask=~valid_mask)
        pooled = masked_mean(encoded, valid_mask)
        return self.head(pooled)
