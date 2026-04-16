import torch
import torch.nn as nn

from .benchmark_blocks import (
    AudioTokenEncoder,
    SequenceClassifierHead,
    VisualTokenEncoder,
    align_audio_to_video_tokens,
    masked_mean,
)


class FusedCrossAttentionTTMModel(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.audio_encoder = AudioTokenEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.visual_encoder = VisualTokenEncoder(hidden_dim=hidden_dim, dropout=dropout)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.visual_temporal = nn.TransformerEncoder(temporal_layer, num_layers=num_layers)
        self.audio_temporal = nn.TransformerEncoder(temporal_layer, num_layers=num_layers)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.head = SequenceClassifierHead(hidden_dim * 2, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        face_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        audio_mask: torch.Tensor = None,
        **_,
    ) -> torch.Tensor:
        visual_tokens = self.visual_encoder(video)
        valid_mask = face_mask & frame_mask
        visual_tokens = self.visual_temporal(visual_tokens, src_key_padding_mask=~valid_mask)

        audio_tokens = self.audio_encoder(audio)
        audio_tokens = self.audio_temporal(audio_tokens)
        audio_tokens = align_audio_to_video_tokens(audio_tokens, visual_tokens.size(1))

        fused_tokens, _ = self.cross_attn(
            query=visual_tokens,
            key=audio_tokens,
            value=audio_tokens,
        )
        fused_tokens = self.norm(visual_tokens + fused_tokens)
        fused_tokens = fused_tokens + self.ffn(fused_tokens)

        pooled_visual = masked_mean(visual_tokens, valid_mask)
        pooled_fused = masked_mean(fused_tokens, valid_mask)
        combined = torch.cat([pooled_visual, pooled_fused], dim=-1)
        return self.head(combined)
