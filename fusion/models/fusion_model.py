import torch
import torch.nn as nn
import torch.nn.functional as F


def align_audio(audio, target_len):
    audio = audio.permute(0, 2, 1)
    audio = F.interpolate(audio, size=target_len, mode="linear", align_corners=False)
    return audio.permute(0, 2, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=500):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        return x + self.pos[:, :x.size(1), :]


class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, k, v, mask=None):
        out, _ = self.attn(q, k, v, key_padding_mask=mask)
        return self.norm(q + out)


class AVFusion(nn.Module):
    def __init__(self, dim_v=768, dim_a=512, dim=512, heads=8):
        super().__init__()

        self.proj_v = nn.Linear(dim_v, dim)
        self.proj_a = nn.Linear(dim_a, dim)

        self.pos_v = PositionalEncoding(dim)
        self.pos_a = PositionalEncoding(dim)

        self.cross_a = CrossAttention(dim, heads)
        self.cross_v = CrossAttention(dim, heads)

        self.fc = nn.Linear(dim * 2, 1)

    def forward(self, visual, audio, mask_v=None, mask_a=None):

        visual = self.proj_v(visual)
        audio = self.proj_a(audio)

        audio = align_audio(audio, visual.size(1))

        if mask_a is not None:
            mask_a = mask_a.float().unsqueeze(1)
            mask_a = F.interpolate(mask_a, size=visual.size(1)).squeeze(1)
            mask_a = mask_a > 0.5

        visual = self.pos_v(visual)
        audio = self.pos_a(audio)

        audio = self.cross_a(audio, visual, visual, mask_v)
        visual = self.cross_v(visual, audio, audio, mask_a)

        fused = torch.cat([audio, visual], dim=-1)

        return self.fc(fused)