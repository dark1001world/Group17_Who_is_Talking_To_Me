import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, dim))

    def forward(self, x):
        T = x.size(1)
        x = x + self.pos_encoding[:, :T, :]
        return self.transformer(x)
