import torch
import torch.nn as nn
from .temporal_encoder import TemporalTransformer

class SpeakerAttributionModel(nn.Module):
    def __init__(self, audio_dim=768, visual_dim=640, fusion_dim=512,
                 num_heads=8, num_temporal_layers=2, dropout=0.1):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.temporal_encoder = TemporalTransformer(fusion_dim, num_layers=num_temporal_layers,
                                                    num_heads=num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim*4, fusion_dim), nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim*2, fusion_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(fusion_dim, 1), nn.Sigmoid()
        )

    def forward(self, audio, visual, track_mask):
        B, T, N, _ = visual.shape
        audio_proj = self.audio_proj(audio)
        visual_proj = self.visual_proj(visual)

        v_reshaped = visual_proj.permute(0,2,1,3).reshape(B*N, T, -1)
        v_encoded = self.temporal_encoder(v_reshaped)
        visual_proj = v_encoded.view(B, N, T, -1).permute(0,2,1,3)

        audio_flat = audio_proj.reshape(B*T, 1, -1)
        visual_flat = visual_proj.reshape(B*T, N, -1)
        mask_flat = track_mask.reshape(B*T, N)

        attn_out, _ = self.cross_attn(query=audio_flat, key=visual_flat, value=visual_flat,
                                      key_padding_mask=~mask_flat)
        audio_enhanced = self.norm(audio_flat + attn_out)
        audio_enhanced = audio_enhanced + self.ffn(audio_enhanced)
        audio_enhanced = audio_enhanced.reshape(B, T, -1)

        audio_expanded = audio_enhanced.unsqueeze(2).expand(-1, -1, N, -1)
        combined = torch.cat([audio_expanded, visual_proj], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        logits = logits.masked_fill(~track_mask, 0.0)
        return logits
