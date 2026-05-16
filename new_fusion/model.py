import torch
import torch.nn as nn

class CrossModalTemporalFusion(nn.Module): # Kept name same so run.py doesn't break
    def __init__(self, audio_dim=512, visual_dim=768, shared_dim=512, dropout=0.5, **kwargs):
        super().__init__()
        
        # 1. Project both to the same dimension
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. The Gate (Decides which modality to trust)
        self.gate = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Sigmoid()
        )
        
        # 3. Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.BatchNorm1d(shared_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim // 2, 2)
        )

    def forward(self, audio_x, visual_x):
        # Flatten just in case they have a dummy seq dimension [Batch, 1, Dim]
        if audio_x.dim() == 3:
            audio_x = audio_x.squeeze(1)
            visual_x = visual_x.squeeze(1)

        # --- MODALITY DROPOUT (Crucial to prevent laziness) ---
        if self.training:
            rand_val = torch.rand(1).item()
            if rand_val < 0.15:
                audio_x = torch.zeros_like(audio_x)
            elif rand_val < 0.30:
                visual_x = torch.zeros_like(visual_x)

        # 1. Project
        a_feat = self.audio_proj(audio_x)
        v_feat = self.visual_proj(visual_x)

        # 2. Calculate Gate 
        # z approaches 1 if model wants audio, 0 if it wants visual
        z = self.gate(torch.cat([a_feat, v_feat], dim=-1))

        # 3. Gated Fusion (Blend them together based on the gate)
        fused = z * a_feat + (1 - z) * v_feat

        # 4. Classify
        logits = self.classifier(fused)
        return logits