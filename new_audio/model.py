import torch
import torch.nn as nn
from transformers import WhisperModel

class WhisperTTM(nn.Module):
    def __init__(
        self, 
        model_size: str = "base", 
        freeze_backbone: bool = True, 
        temporal_depth: int = 2,
        dropout: float = 0.3
    ):
        super(WhisperTTM, self).__init__()
        
        # 1. Load the powerful semantic encoder from Hugging Face
        # "base" is 74M parameters - perfect for fast, frozen training
        hf_name = f"openai/whisper-{model_size}"
        print(f"[WhisperTTM] Loading {hf_name} encoder...")
        self.encoder = WhisperModel.from_pretrained(hf_name).encoder
        
        # Determine the dimension dynamically based on the model size (base = 512)
        self.embed_dim = self.encoder.config.d_model
        
        # 2. Freeze the 74M semantic parameters
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[WhisperTTM] Whisper backbone successfully frozen.")

        # 3. Modern, Flash-Attention optimized Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True # Keeps shapes as [Batch, Time, Dim] for sanity
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=temporal_depth
        )
        
        # 4. Dense Prediction Head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim // 2, 2) # Binary TTM Output
        )

    def forward(self, mel_spectrogram: torch.Tensor, track: torch.Tensor = None):
        """
        Expected Input:
        mel_spectrogram: [Batch, 80, Time] - The standard Whisper Mel format.
                         (It does not have to be 3000 frames long! 
                         You can pass 500 frames for a 5-second clip).
        track: Ignored (kept for engine compatibility to avoid per-batch TypeError).
        """
        # 1. Whisper Feature Extraction (Frozen)
        # Note: WhisperModel from HF expects [Batch, 80, Time]
        encoder_outputs = self.encoder(mel_spectrogram)
        
        # Hidden states shape: [Batch, Time_compressed, Embed_Dim]
        # Whisper naturally compresses the time dimension by a factor of 2
        x = encoder_outputs.last_hidden_state 
        
        # 2. Temporal Context Building (Trainable)
        # Allows frames to talk to each other to figure out conversational context
        x = self.temporal_transformer(x)
        
        # 3. Global Temporal Pooling
        # Instead of their massive AdaptiveAvgPool, we do a clean mean pool across the time dimension
        x = x.mean(dim=1) # Shape collapses to [Batch, Embed_Dim]
        
        # 4. Final Classification
        logits = self.classifier(x) # Shape: [Batch, 2]
        
        return logits