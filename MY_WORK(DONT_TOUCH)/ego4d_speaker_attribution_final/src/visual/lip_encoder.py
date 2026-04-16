import torch
import torch.nn as nn
import torch.nn.functional as F

class LipMotionEncoder(nn.Module):
    def __init__(self, input_channels=3, temporal_window=5, output_dim=128):
        super().__init__()
        self.temporal_window = temporal_window
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, mouth_sequence):
        B, T, C, H, W = mouth_sequence.shape
        if T != self.temporal_window:
            if T > self.temporal_window:
                mouth_sequence = mouth_sequence[:, :self.temporal_window]
            else:
                pad = self.temporal_window - T
                mouth_sequence = F.pad(mouth_sequence, (0,0,0,0,0,0,0,pad))
        x = mouth_sequence.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        x = x.view(B, -1)
        return self.fc(x)
