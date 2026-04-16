"""
audio_model.py
──────────────
Whisper-Large-V3 encoder fine-tuned for the TTM task.

Architecture
────────────
  WhisperEncoder (frozen first N layers)
      │
      ▼ hidden states (B, T_enc, 1280)      T_enc = 1500 for 30 s audio
      │
  TemporalProjection  (1280 → projection_dim)
      │
      ▼ embeddings  (B, T_enc, 512)         ← what the cross-transformer will consume
      │
  TTMHead  (linear classifier, optional)
      │
      ▼ logits  (B, 2)  or  (B, T_enc, 2)  for clip-level / frame-level
"""

from __future__ import annotations
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperConfig
from transformers.modeling_outputs import BaseModelOutput

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

class TemporalProjection(nn.Module):
    """
    Projects Whisper encoder hidden states from 1280-d → projection_dim.
    Uses a 2-layer MLP with LayerNorm and GELU – lightweight but expressive.
    """

    
    def __init__(self, in_dim: int = 1280, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim), # Keep this
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_dim)  →  (B, T, out_dim)"""
        return self.net(x)


class TTMHead(nn.Module):
    """
    Binary classification head for TTM.
    Supports both clip-level (mean pool) and frame-level predictions.
    """

    def __init__(self, embed_dim: int = 512, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.clip_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self.frame_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(
        self, embeddings: torch.Tensor, pool: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        embeddings : (B, T, D)
        Returns
        -------
        clip_logits  : (B, num_classes)
        frame_logits : (B, T, num_classes)
        """
        frame_logits = self.frame_head(embeddings)               # (B, T, C)
        pooled       = embeddings.mean(dim=1)                    # (B, D)
        clip_logits  = self.clip_head(pooled)                    # (B, C)
        return clip_logits, frame_logits


# ── main model ────────────────────────────────────────────────────────────────

class WhisperTTM(nn.Module):
    """
    Whisper-Large-V3 encoder fine-tuned for TTM.

    Parameters
    ----------
    model_name           : HuggingFace model id
    freeze_encoder_layers: how many of the 32 encoder layers to freeze
    projection_dim       : output embedding dimension (fed to cross-transformer)
    dropout              : dropout applied in projection + head
    num_classes          : 2 for binary TTM
    gradient_checkpointing: saves ~40 % VRAM during training
    """

    ENCODER_HIDDEN = 1280   # Whisper Large V3 hidden size

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        freeze_encoder_layers: int = 20,
        projection_dim: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        # Load only encoder weights (no decoder – saves ~1 GB VRAM)
        logger.info("Loading Whisper encoder from '%s' …", model_name)
        full_model = WhisperModel.from_pretrained(model_name)
        self.encoder = full_model.encoder
        del full_model  # free decoder weights immediately

        # Gradient checkpointing
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # Freeze first N encoder layers
        self._freeze_encoder_layers(freeze_encoder_layers)

        # Projection + classification head
        self.projection = TemporalProjection(self.ENCODER_HIDDEN, projection_dim, dropout)
        self.head       = TTMHead(projection_dim, num_classes, dropout)

        logger.info(
            "Model ready. Trainable params: %s  |  Total params: %s",
            f"{self._count_params(trainable=True):,}",
            f"{self._count_params(trainable=False):,}",
        )

    # ── freeze helpers ───────────────────────────────────────────────────────

    def _freeze_encoder_layers(self, n: int):
        """Freeze CNN conv layers + first n transformer blocks."""
        # Always freeze conv stem
        for p in self.encoder.conv1.parameters():
            p.requires_grad = False
        for p in self.encoder.conv2.parameters():
            p.requires_grad = False
        for p in self.encoder.embed_positions.parameters():
            p.requires_grad = False

        # Freeze first n transformer layers
        for i, layer in enumerate(self.encoder.layers):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False

        frozen = sum(1 for p in self.encoder.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.encoder.parameters() if p.requires_grad)
        logger.info(
            "Encoder: %d frozen layers, %d trainable params (encoder only)",
            n, sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )

    def _count_params(self, trainable: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad == trainable
        )

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_features: torch.Tensor,          # (B, 80, 3000)
        return_embeddings: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with keys:
          'clip_logits'   : (B, num_classes)
          'frame_logits'  : (B, T_enc, num_classes)
          'embeddings'    : (B, T_enc, projection_dim)  – always returned
          'hidden_states' : tuple of all encoder hidden states (only if output_hidden_states=True)
        """
        enc_out: BaseModelOutput = self.encoder(
            input_features,
            output_hidden_states=output_hidden_states,
        )

        hidden = enc_out.last_hidden_state          # (B, T_enc, 1280)
        embeddings = self.projection(hidden)        # (B, T_enc, proj_dim)
        clip_logits, frame_logits = self.head(embeddings)

        result = {
            "clip_logits":  clip_logits,
            "frame_logits": frame_logits,
            "embeddings":   embeddings,
        }
        if output_hidden_states:
            result["hidden_states"] = enc_out.hidden_states

        return result

    # ── convenience: encoder-only (no grad) for embedding extraction ─────────

    @torch.no_grad()
    def encode(
        self,
        input_features: torch.Tensor,
        layers_to_avg: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Extract projected embeddings without computing gradients.
        layers_to_avg: list of encoder layer indices to average (e.g. [-4,-3,-2,-1]).
                       If None, uses last_hidden_state only.
        Returns: (B, T_enc, projection_dim)
        """
        enc_out = self.encoder(
            input_features,
            output_hidden_states=(layers_to_avg is not None),
        )

        if layers_to_avg is not None:
            all_hs = enc_out.hidden_states       # tuple of (B, T, 1280) x 33
            selected = torch.stack(
                [all_hs[i] for i in layers_to_avg], dim=0
            ).mean(dim=0)                        # (B, T, 1280)
        else:
            selected = enc_out.last_hidden_state

        return self.projection(selected)         # (B, T_enc, proj_dim)


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2, reduction="mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Use log_softmax + nll for numerical stability instead of CE then exp
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)   # (N, C)
        probs     = torch.exp(log_probs)                               # (N, C)

        # Gather the log-prob and prob for the true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        pt     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # (N,)

        focal  = -self.alpha * (1 - pt).pow(self.gamma) * log_pt       # (N,)

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def build_loss(cfg: dict) -> nn.Module:
    if cfg["training"].get("use_focal_loss", True):
        return FocalLoss(
            alpha=cfg["training"]["focal_alpha"],
            gamma=cfg["training"]["focal_gamma"],
        )
    return nn.CrossEntropyLoss()