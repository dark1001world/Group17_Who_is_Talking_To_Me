import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(sequence.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (sequence * mask).sum(dim=1) / denom


class AudioTokenEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=6, stride=3, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.proj(x)


class VisualTokenEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        backbone = resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = video.shape
        flat = video.reshape(batch_size * time_steps, channels, height, width)
        features = self.backbone(flat)
        features = self.proj(features)
        return features.view(batch_size, time_steps, -1)


class SequenceClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def align_audio_to_video_tokens(audio_tokens: torch.Tensor, target_len: int) -> torch.Tensor:
    if audio_tokens.size(1) == target_len:
        return audio_tokens
    x = audio_tokens.transpose(1, 2)
    x = F.adaptive_avg_pool1d(x, target_len)
    return x.transpose(1, 2)
