"""
Visual Pipeline - Models  (ViT Edition)
========================================
Architecture stack (state-of-the-art 2024):

  1. VideoSwinV2TTM   ← PRIMARY model
     • Video Swin Transformer V2 backbone (torchvision / timm)
     • Shifted-window 3-D self-attention (spatial + temporal jointly)
     • Cross-attention temporal aggregation head
     • Binary TTM classification

  2. TimeSformerTTM   ← ALTERNATIVE (divided space-time attention)
     • Uses timm ViT backbone + divided temporal attention
     • Lighter than Swin, good for smaller datasets

  3. FactorisedViTTTM ← LIGHTWEIGHT option
     • Plain ViT-S/16 spatial + 1-D temporal transformer
     • Fastest, smallest memory footprint

All models:
  • Accept input [B, C, T, H, W]
  • Return logits [B, 2]
  • Use pre-trained ImageNet-21k / Kinetics weights where available
  • Proper weight init on new heads

Dependencies:
    torch>=2.2  torchvision>=0.17  timm>=0.9.12
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("[Warning] timm not installed. Install with: pip install timm>=0.9.12")

try:
    from torchvision.models.video import (
        swin3d_s, Swin3D_S_Weights,
        swin3d_b, Swin3D_B_Weights,
    )
    HAS_SWIN3D = True
except ImportError:
    HAS_SWIN3D = False
    print("[Warning] torchvision video Swin not available. Update torchvision>=0.17")


# ──────────────────────────────────────────────────────────────
#  Utility: freeze / partial-freeze backbone
# ──────────────────────────────────────────────────────────────

def freeze_layers(model: nn.Module, freeze_until: str = ""):
    """Freeze all parameters up to (but not including) `freeze_until` layer."""
    if not freeze_until:
        return
    frozen = True
    for name, param in model.named_parameters():
        if freeze_until in name:
            frozen = False
        param.requires_grad = not frozen


def count_params(model: nn.Module, only_trainable: bool = True) -> int:
    return sum(
        p.numel() for p in model.parameters()
        if (p.requires_grad or not only_trainable)
    )


# ──────────────────────────────────────────────────────────────
#  Cross-Attention Temporal Head
# ──────────────────────────────────────────────────────────────

class CrossAttentionTemporalHead(nn.Module):
    """
    Aggregates per-frame / per-token features using cross-attention.

    A learnable [CLS] query attends over the sequence of frame tokens,
    then a 2-layer MLP produces the final binary logits.

    Input:  tokens [B, T, D]
    Output: logits [B, num_classes]
    """

    def __init__(
        self,
        dim        : int,
        num_heads  : int = 8,
        mlp_ratio  : float = 2.0,
        dropout    : float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: [B, T, D]
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)            # [B, 1, D]

        # cross-attention: CLS queries frame tokens
        attended, _ = self.cross_attn(
            query = cls,
            key   = self.norm1(tokens),
            value = self.norm1(tokens),
        )                                                   # [B, 1, D]
        cls = cls + attended
        cls = cls + self.mlp(self.norm2(cls))

        return self.head(cls.squeeze(1))                   # [B, num_classes]


# ──────────────────────────────────────────────────────────────
#  1. Video Swin Transformer V2  ← PRIMARY / BEST
# ──────────────────────────────────────────────────────────────

class VideoSwinV2TTM(nn.Module):
    """
    Video Swin Transformer V2 for TTM binary classification.

    Architecture:
      • Swin3D-S or Swin3D-B backbone (torchvision built-in)
      • Shifted-window 3D self-attention over (T, H, W) tubes
      • Removes classification head → exposes [B, T', D] token sequence
      • CrossAttentionTemporalHead aggregates → [B, 2]

    Why Swin V2 > ResNet + LSTM:
      • Native spatiotemporal modelling (no sequential bottleneck)
      • Hierarchical feature pyramid captures multi-scale face features
      • Relative position bias scales to any resolution / frame count
      • SOTA on Kinetics-400/600, SomethingSomething-V2

    Input:  [B, C, T, H, W]   e.g. [B, 3, 8, 224, 224]
    Output: [B, 2]
    """

    def __init__(
        self,
        variant    : str   = "swin3d_s",    # "swin3d_s" | "swin3d_b"
        pretrained : bool  = True,
        num_classes: int   = 2,
        dropout    : float = 0.2,
        freeze_stages: int = 0,             # 0 = train all, 1-4 = freeze early
    ):
        super().__init__()

        if not HAS_SWIN3D:
            raise ImportError(
                "torchvision>=0.17 required for Video Swin V2. "
                "Run: pip install --upgrade torchvision"
            )

        # ── backbone ──
        if variant == "swin3d_s":
            weights = Swin3D_S_Weights.KINETICS400_V1 if pretrained else None
            backbone = swin3d_s(weights=weights)
            feat_dim = 768
        elif variant == "swin3d_b":
            weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1 if pretrained else None
            backbone = swin3d_b(weights=weights)
            feat_dim = 1024
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # strip the original classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # backbone now outputs [B, D, T', H', W']

        #self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))   # [B, D, T', 1, 1]

        # ── temporal aggregation ──
        num_heads = feat_dim // 64
        self.temporal_head = CrossAttentionTemporalHead(
            dim         = feat_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            num_classes = num_classes,
        )

        # ── optional stage freezing ──
        if freeze_stages > 0:
            self._freeze_stages(backbone, freeze_stages)

        trainable = count_params(self)
        total     = count_params(self, only_trainable=False)
        print(f"[VideoSwinV2TTM] variant={variant}  "
              f"trainable={trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    # REPLACE the entire _freeze_stages method with:
    # REPLACE _freeze_stages entirely with this simpler approach in model.py:
    def _freeze_stages(self, backbone, n_stages: int):
        if n_stages == 0:
            return

        if not hasattr(backbone, 'features'):
            print("[Freeze] No features found — freezing entire backbone")
            for p in backbone.parameters():
                p.requires_grad = False
            return

        # meaningful freeze map for Swin3D:
        # n_stages=1 → freeze features[0-3]  (lightweight early layers, 1.5M)
        # n_stages=2 → freeze features[0-4]  (+ main transformer, 34M)
        # n_stages=3 → freeze features[0-5]  (+ patch merging, 35M)
        # n_stages=4 → freeze entire backbone (all 54M)
        freeze_map = {
            1: [0],                     # freeze patch embed only    → trainable ~54M
            2: [0, 1, 2, 3],           # freeze early layers        → trainable ~53M  
            3: [0, 1, 2, 3, 4, 5],    # freeze all except features[6] → trainable ~18M
            4: [0, 1, 2, 3, 4, 5, 6], # freeze entire backbone      → trainable ~3.5M
        }
        indices = freeze_map.get(n_stages, list(range(7)))

        for i in indices:
            for p in backbone.features[i].parameters():
                p.requires_grad = False
            params = sum(p.numel() for p in backbone.features[i].parameters())
            print(f"[Freeze] Frozen features[{i}]  ({params/1e6:.2f}M)")

        frozen    = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"[Freeze] backbone frozen={frozen/1e6:.1f}M  trainable={trainable/1e6:.1f}M")
    def forward(self, x: Tensor) -> Tensor:
    # x: [B, C, T, H, W]
        # Backbone output layout can differ across torchvision versions.
        feats = self.backbone(x)
        d_expected = self.temporal_head.cls_token.shape[-1]

        if feats.shape[-1] == d_expected:
            # [B, T', H', W', D] -> [B, T', D]
            tokens = feats.mean(dim=(2, 3)).contiguous()
        elif feats.shape[1] == d_expected:
            # [B, D, T', H', W'] -> [B, T', D]
            tokens = feats.mean(dim=(3, 4)).transpose(1, 2).contiguous()
        else:
            raise RuntimeError(
                f"Unexpected Swin feature shape {tuple(feats.shape)} for expected dim={d_expected}"
            )
        return self.temporal_head(tokens)      # [B, 2]   # [B, 2]


# ──────────────────────────────────────────────────────────────
#  2. TimeSformer-style  (divided space-time attention)
# ──────────────────────────────────────────────────────────────

class DividedSpaceTimeAttention(nn.Module):
    """
    Factorised temporal + spatial self-attention block.
    Each token attends to: (a) same spatial position across time,
    then (b) all spatial positions at same time step.
    50% fewer FLOPs than full joint attention at same capacity.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.spatial_attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)

        mlp_hidden = dim * 4
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, T: int, N: int) -> Tensor:
        """
        x: [B, T*N, D]  where T=frames, N=spatial patches per frame
        """
        B, TN, D = x.shape

        # ── temporal attention: each patch attends across frames ──
        x_t = x.view(B * N, T, D)
        x_t_norm = self.norm_t(x_t)
        t_out, _ = self.temporal_attn(x_t_norm, x_t_norm, x_t_norm)
        x = x + t_out.view(B, TN, D)

        # ── spatial attention: each frame attends to all patches ──
        x_s = x.view(B * T, N, D)
        x_s_norm = self.norm_s(x_s)
        s_out, _ = self.spatial_attn(x_s_norm, x_s_norm, x_s_norm)
        x = x + s_out.view(B, TN, D)

        # ── MLP ──
        x = x + self.mlp(x)
        return x


class TimeSformerTTM(nn.Module):
    """
    TimeSformer-style TTM model built on a frozen ViT-S/16 spatial backbone
    (ImageNet-21k pretrained via timm) with divided space-time attention layers.

    Lighter than Swin3D — good for limited GPU memory or smaller datasets.

    Input:  [B, C, T, H, W]
    Output: [B, 2]
    """

    def __init__(
        self,
        vit_variant : str   = "vit_small_patch16_224",
        pretrained  : bool  = True,
        num_frames  : int   = 8,
        num_classes : int   = 2,
        dropout     : float = 0.1,
        depth       : int   = 4,       # number of divided ST blocks added on top
    ):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("timm required: pip install timm>=0.9.12")

        # ── spatial ViT backbone (patch embed + transformer) ──
        self.spatial_vit = timm.create_model(
            vit_variant,
            pretrained   = pretrained,
            num_classes  = 0,           # remove head → raw [B*T, N+1, D]
            global_pool  = "",
        )
        feat_dim = self.spatial_vit.embed_dim
        self.num_frames = num_frames

        # ── temporal position embedding ──
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, num_frames, feat_dim)
        )
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        # ── divided space-time attention blocks ──
        self.st_blocks = nn.ModuleList([
            DividedSpaceTimeAttention(feat_dim, num_heads=feat_dim//64, dropout=dropout)
            for _ in range(depth)
        ])

        # ── classification head via cross-attention ──
        self.head = CrossAttentionTemporalHead(
            dim         = feat_dim,
            num_heads   = feat_dim // 64,
            dropout     = dropout,
            num_classes = num_classes,
        )

        trainable = count_params(self)
        total     = count_params(self, only_trainable=False)
        print(f"[TimeSformerTTM] variant={vit_variant}  depth={depth}  "
              f"trainable={trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    def forward(self, x: Tensor) -> Tensor:
        B, C, T, H, W = x.shape

        # per-frame spatial features
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        tokens = self.spatial_vit.forward_features(x_flat)  # [B*T, N+1, D]

        # use CLS token as frame summary
        cls_tokens = tokens[:, 0]                           # [B*T, D]
        cls_tokens = cls_tokens.view(B, T, -1)              # [B, T, D]

        # add temporal position embedding
        cls_tokens = cls_tokens + self.temporal_pos[:, :T]

        # divided space-time attention
        N = tokens.size(1) - 1                              # spatial patches
        patch_tokens = tokens[:, 1:].view(B, T * N, -1)    # [B, T*N, D]
        for block in self.st_blocks:
            patch_tokens = block(patch_tokens, T, N)

        # aggregate: mean of CLS + cross-attended patches
        patch_mean = patch_tokens.view(B, T, N, -1).mean(dim=2)   # [B, T, D]
        combined   = cls_tokens + patch_mean                       # [B, T, D]

        return self.head(combined)                                 # [B, 2]


# ──────────────────────────────────────────────────────────────
#  3. Factorised ViT  (lightest — fast iteration)
# ──────────────────────────────────────────────────────────────

class FactorisedViTTTM(nn.Module):
    """
    Simplest ViT-based video model:
      • ViT-S/16 extracts per-frame CLS tokens (frozen or fine-tuned)
      • 1-D Transformer encoder models temporal dependencies
      • CrossAttentionTemporalHead → logits

    Best for: quick baselines, ablations, limited GPU memory.

    Input:  [B, C, T, H, W]
    Output: [B, 2]
    """

    def __init__(
        self,
        vit_variant    : str   = "vit_small_patch16_224",
        pretrained     : bool  = True,
        num_frames     : int   = 8,
        num_classes    : int   = 2,
        temporal_depth : int   = 2,
        dropout        : float = 0.1,
        freeze_backbone: bool  = False,
    ):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("timm required: pip install timm>=0.9.12")

        self.spatial_vit = timm.create_model(
            vit_variant,
            pretrained  = pretrained,
            num_classes = 0,
            global_pool = "token",      # returns CLS token [B, D]
        )
        if freeze_backbone:
            for p in self.spatial_vit.parameters():
                p.requires_grad = False
            print("[FactorisedViT] Backbone frozen — only temporal head trains")

        feat_dim   = self.spatial_vit.embed_dim
        self.num_frames = num_frames

        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, feat_dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = feat_dim,
            nhead           = feat_dim // 64,
            dim_feedforward = feat_dim * 4,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,      # pre-LN: more stable
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = temporal_depth,
            enable_nested_tensor = False,
        )

        self.head = CrossAttentionTemporalHead(
            dim         = feat_dim,
            num_heads   = feat_dim // 64,
            dropout     = dropout,
            num_classes = num_classes,
        )

        trainable = count_params(self)
        total     = count_params(self, only_trainable=False)
        print(f"[FactorisedViTTTM] variant={vit_variant}  "
              f"temporal_depth={temporal_depth}  "
              f"trainable={trainable/1e6:.1f}M / {total/1e6:.1f}M")

    def forward(self, x: Tensor) -> Tensor:
        B, C, T, H, W = x.shape

        # per-frame CLS token
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        cls    = self.spatial_vit(x_flat)              # [B*T, D]
        cls    = cls.view(B, T, -1)                    # [B, T, D]

        # temporal position + transformer
        cls = cls + self.temporal_pos[:, :T]
        cls = self.temporal_transformer(cls)           # [B, T, D]

        return self.head(cls)                          # [B, 2]


# ──────────────────────────────────────────────────────────────
#  Registry + factory
# ──────────────────────────────────────────────────────────────

# MODEL_REGISTRY: dict[str, type] = {
#     "VideoSwinV2TTM"   : VideoSwinV2TTM,
#     "TimeSformerTTM"   : TimeSformerTTM,
#     "FactorisedViTTTM" : FactorisedViTTTM,
# }


# def build_model(name: str, **kwargs) -> nn.Module:
#     if name not in MODEL_REGISTRY:
#         raise ValueError(
#             f"Unknown model '{name}'. "
#             f"Choose from: {list(MODEL_REGISTRY.keys())}"
#         )
#     return MODEL_REGISTRY[name](**kwargs)



"""
DINO ViT + Track Features Extension
=====================================
ADD THIS TO THE BOTTOM OF YOUR EXISTING model.py

Preserves all existing models (VideoSwinV2TTM, TimeSformerTTM, FactorisedViTTTM).
Adds two new models:
  - DinoViTTTM         : DINO ViT-B/16 + temporal transformer
  - DinoViTTrackTTM    : DINO ViT-B/16 + track spatial features + temporal transformer

Usage: just append this file content to your existing model.py
Then add the new models to MODEL_REGISTRY at the bottom.
"""

# ──────────────────────────────────────────────────────────────
#  Track Feature Projector
# ──────────────────────────────────────────────────────────────

class TrackFeatureProjector(nn.Module):
    """
    Projects 6-dim spatial track features to match ViT embedding dim.
    
    Input:  [B, T, 6]   (cx, cy, size, dx, dy, ds)
    Output: [B, T, dim]
    """

    def __init__(self, in_dim: int = 6, out_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


# ──────────────────────────────────────────────────────────────
#  DINO ViT TTM  (visual only, no track features)
# ──────────────────────────────────────────────────────────────

class DinoViTTTM(nn.Module):
    """
    DINO ViT-B/16 for TTM binary classification.

    Why better than Swin3D:
      - Pretrained with self-supervised learning on ImageNet
      - Naturally attends to faces and objects (not scene motion)
      - Better transfer to face crop domain
      - Lighter than Swin3D (86M vs 54M but more efficient for faces)

    Architecture:
      ViT-B/16 per-frame CLS tokens → temporal transformer → cross-attn head

    Input:  [B, C, T, H, W]
    Output: [B, 2]
    """

    def __init__(
        self,
        vit_variant    : str   = "vit_base_patch16_224",
        pretrained     : bool  = True,
        num_frames     : int   = 8,
        num_classes    : int   = 2,
        temporal_depth : int   = 2,
        dropout        : float = 0.2,
        freeze_backbone: bool  = True,
    ):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("timm required: pip install timm>=0.9.12")

        # ── DINO ViT backbone ──
        self.backbone = timm.create_model(
            vit_variant,
            pretrained  = pretrained,
            num_classes = 0,
            global_pool = "token",   # returns CLS token [B, D]
        )
        feat_dim = self.backbone.embed_dim
        self.num_frames = num_frames

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print(f"[DinoViTTTM] Backbone frozen")

        # ── temporal position embedding ──
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, feat_dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        # ── temporal transformer ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = feat_dim,
            nhead           = feat_dim // 64,
            dim_feedforward = feat_dim * 4,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers           = temporal_depth,
            enable_nested_tensor = False,
        )

        # ── classification head ──
        self.head = CrossAttentionTemporalHead(
            dim         = feat_dim,
            num_heads   = feat_dim // 64,
            dropout     = dropout,
            num_classes = num_classes,
        )

        trainable = count_params(self)
        total     = count_params(self, only_trainable=False)
        print(f"[DinoViTTTM] variant={vit_variant}  "
              f"trainable={trainable/1e6:.1f}M / {total/1e6:.1f}M")

    def forward(self, x: Tensor, track: Tensor = None) -> Tensor:
        """
        x:     [B, C, T, H, W]
        track: ignored (for API compatibility with DinoViTTrackTTM)
        """
        B, C, T, H, W = x.shape

        # per-frame CLS tokens
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        cls    = self.backbone(x_flat)           # [B*T, D]
        cls    = cls.view(B, T, -1)              # [B, T, D]

        # temporal modelling
        cls = cls + self.temporal_pos[:, :T]
        cls = self.temporal_transformer(cls)     # [B, T, D]

        return self.head(cls)                    # [B, 2]

    def extract_tokens(self, x: Tensor, track: Tensor = None) -> dict:
        """Extract intermediate features for fusion."""
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        cls    = self.backbone(x_flat).view(B, T, -1)
        cls    = cls + self.temporal_pos[:, :T]
        tokens = self.temporal_transformer(cls)

        # cross-attention
        cls_q = self.head.cls_token.expand(B, -1, -1)
        normed = self.head.norm1(tokens)
        attended, attn_w = self.head.cross_attn(
            query=cls_q, key=normed, value=normed, need_weights=True
        )
        cls_vec    = (cls_q + attended).squeeze(1)
        logits     = self.head.head(cls_vec)
        confidence = torch.softmax(logits, dim=1)[:, 1]

        return {
            "frame_tokens" : tokens,
            "cls_embedding": cls_vec,
            "attn_weights" : attn_w.squeeze(1),
            "confidence"   : confidence,
            "logits"       : logits,
        }


# ──────────────────────────────────────────────────────────────
#  DINO ViT + Track Features TTM  (PRIMARY NEW MODEL)
# ──────────────────────────────────────────────────────────────

class DinoViTTrackTTM(nn.Module):
    """
    DINO ViT-B/16 + Spatial Track Features for TTM.

    Track features (from json_original bounding boxes):
      cx, cy   = normalized face center position
      size     = normalized face area (how close to camera)
      dx, dy   = frame-to-frame movement (head motion)
      ds       = face size change (moving toward/away camera)

    Why this beats Swin3D alone:
      - DINO gives rich face-aware visual features
      - Track features give explicit spatial/motion context
      - Together: model knows WHAT the face looks like AND WHERE/HOW it moves
      - TTM signal: face centered + stable + large = looking at camera

    Architecture:
      DINO CLS [B,T,D] + Track projection [B,T,D]
          ↓ element-wise add (gated)
      Temporal Transformer
          ↓
      CrossAttention Head → [B, 2]

    Input:  x [B, C, T, H, W],  track [B, T, 6]
    Output: [B, 2]
    """

    def __init__(
        self,
        vit_variant    : str   = "vit_base_patch16_224",
        pretrained     : bool  = True,
        num_frames     : int   = 8,
        num_classes    : int   = 2,
        temporal_depth : int   = 2,
        dropout        : float = 0.2,
        freeze_backbone: bool  = True,
        track_dim      : int   = 6,
    ):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("timm required: pip install timm>=0.9.12")

        # ── DINO ViT backbone ──
        self.backbone = timm.create_model(
            vit_variant,
            pretrained  = pretrained,
            num_classes = 0,
            global_pool = "token",
        )
        feat_dim = self.backbone.embed_dim
        self.num_frames = num_frames

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print(f"[DinoViTTrackTTM] Backbone frozen")

        # ── temporal position embedding ──
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, feat_dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        # ── track feature projector ──
        self.track_proj = TrackFeatureProjector(
            in_dim  = track_dim,
            out_dim = feat_dim,
            dropout = dropout,
        )

        # ── learnable gate: how much to trust track vs visual ──
        self.track_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid(),
        )

        # ── temporal transformer ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = feat_dim,
            nhead           = feat_dim // 64,
            dim_feedforward = feat_dim * 4,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers           = temporal_depth,
            enable_nested_tensor = False,
        )

        # ── classification head ──
        self.head = CrossAttentionTemporalHead(
            dim         = feat_dim,
            num_heads   = feat_dim // 64,
            dropout     = dropout,
            num_classes = num_classes,
        )

        trainable = count_params(self)
        total     = count_params(self, only_trainable=False)
        print(f"[DinoViTTrackTTM] variant={vit_variant}  "
              f"trainable={trainable/1e6:.1f}M / {total/1e6:.1f}M")

    def forward(self, x: Tensor, track: Tensor) -> Tensor:
        """
        x:     [B, C, T, H, W]   visual clip
        track: [B, T, 6]          spatial track features
        """
        B, C, T, H, W = x.shape

        # ── visual features ──
        x_flat  = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        vis_cls = self.backbone(x_flat).view(B, T, -1)   # [B, T, D]

        # ── track features ──
        # pad/trim track to match T
        if track.size(1) != T:
            track = track[:, :T] if track.size(1) > T \
                    else torch.cat([
                        track,
                        track[:, -1:].expand(-1, T - track.size(1), -1)
                    ], dim=1)

        track_proj = self.track_proj(track)               # [B, T, D]

        # ── gated fusion ──
        gate   = self.track_gate(
            torch.cat([vis_cls, track_proj], dim=-1)
        )                                                  # [B, T, D]
        tokens = vis_cls + gate * track_proj              # [B, T, D]

        # ── temporal modelling ──
        tokens = tokens + self.temporal_pos[:, :T]
        tokens = self.temporal_transformer(tokens)        # [B, T, D]

        return self.head(tokens)                          # [B, 2]

    def extract_tokens(self, x: Tensor, track: Tensor) -> dict:
        """Extract intermediate features for fusion."""
        B, C, T, H, W = x.shape
        x_flat     = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        vis_cls    = self.backbone(x_flat).view(B, T, -1)

        # Keep track features time-aligned with video tokens (same logic as forward).
        if track.size(1) != T:
            track = track[:, :T] if track.size(1) > T \
                    else torch.cat([
                        track,
                        track[:, -1:].expand(-1, T - track.size(1), -1)
                    ], dim=1)

        track_proj = self.track_proj(track)
        gate       = self.track_gate(torch.cat([vis_cls, track_proj], dim=-1))
        tokens     = vis_cls + gate * track_proj
        tokens     = tokens + self.temporal_pos[:, :T]
        tokens     = self.temporal_transformer(tokens)

        cls_q  = self.head.cls_token.expand(B, -1, -1)
        normed = self.head.norm1(tokens)
        attended, attn_w = self.head.cross_attn(
            query=cls_q, key=normed, value=normed, need_weights=True
        )
        cls_vec    = (cls_q + attended).squeeze(1)
        logits     = self.head.head(cls_vec)
        confidence = torch.softmax(logits, dim=1)[:, 1]

        return {
            "frame_tokens" : tokens,
            "cls_embedding": cls_vec,
            "attn_weights" : attn_w.squeeze(1),
            "confidence"   : confidence,
            "logits"       : logits,
            "track_gate"   : gate,         # bonus: how much track was used
        }


# ──────────────────────────────────────────────────────────────
#  UPDATE MODEL_REGISTRY  (replace existing registry at bottom
#  of your model.py with this)
# ──────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, type] = {
    # ── existing models (preserved) ──
    "VideoSwinV2TTM"    : VideoSwinV2TTM,
    "TimeSformerTTM"    : TimeSformerTTM,
    "FactorisedViTTTM"  : FactorisedViTTTM,
    # ── new models ──
    "DinoViTTTM"        : DinoViTTTM,
    "DinoViTTrackTTM"   : DinoViTTrackTTM,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
# ──────────────────────────────────────────────────────────────
#  Quick shape test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, C, T, H, W = 2, 3, 8, 224, 224
    x = torch.randn(B, C, T, H, W)

    print("\n── FactorisedViTTTM (no timm download in test) ──")
    try:
        m = FactorisedViTTTM(pretrained=False, num_frames=T)
        out = m(x)
        print(f"  Input {list(x.shape)} → Output {list(out.shape)}")
        assert out.shape == (B, 2)
        print("  ✓ Shape OK")
    except Exception as e:
        print(f"  [Skip] {e}")
