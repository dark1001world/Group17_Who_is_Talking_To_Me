from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from data_loader import ViTImagerLoader
from model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export fusion-ready .pt embeddings from trained visual model"
    )

    # Data paths
    p.add_argument("--source_path", required=True)
    p.add_argument("--json_path", required=True)
    p.add_argument("--gt_path", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file", required=True)

    # Model/checkpoint
    p.add_argument(
        "--model",
        default="DinoViTTrackTTM",
        choices=["DinoViTTTM", "DinoViTTrackTTM"],
    )
    p.add_argument("--variant", default="vit_base_patch16_224")
    p.add_argument("--clip_frames", type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--temporal_depth", type=int, default=2)
    p.add_argument("--freeze_backbone", action="store_true", default=True)
    p.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone")
    p.add_argument("--checkpoint", required=True)

    # Export setup
    p.add_argument(
        "--output_root",
        default="/DATA/G17/Group17_Who_is_Talking_To_Me/new_visual/embeddings2",
        help="Will create train/ and val/ subfolders with fusion-ready .pt files",
    )
    p.add_argument(
        "--audio_embed_root",
        default="/DATA/G17/outputs/ttm_audio/embeddings",
        help="Optional root containing split/uid.pt audio features. Missing files fall back to zeros.",
    )
    p.add_argument("--audio_dim", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1, help="Reserved for future batching; kept at 1")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--min_seg_frames", type=int, default=8)
    p.add_argument("--use_track", action="store_true", default=True)
    p.add_argument("--no_use_track", action="store_false", dest="use_track")
    p.add_argument("--cpu", action="store_true")

    return p.parse_args()


def build_visual_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    kwargs = dict(
        pretrained=True,
        num_classes=2,
        dropout=args.dropout,
        vit_variant=args.variant,
        num_frames=args.clip_frames,
        temporal_depth=args.temporal_depth,
        freeze_backbone=args.freeze_backbone,
    )

    model = build_model(args.model, **kwargs)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[Load] checkpoint={args.checkpoint} missing={len(missing)} unexpected={len(unexpected)}"
    )

    if not hasattr(model, "extract_tokens"):
        raise RuntimeError(
            f"Model {args.model} does not expose extract_tokens(); use DinoViTTTM or DinoViTTrackTTM"
        )

    model.to(device).eval()
    return model


def _to_2d_float_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        t = value.detach().cpu().float()
    else:
        try:
            t = torch.as_tensor(value).float()
        except Exception:
            return None

    if t.ndim == 1:
        return t.unsqueeze(0)
    if t.ndim == 2:
        return t
    if t.ndim >= 3:
        # Collapse leading dimensions, keep last dim as feature dim.
        return t.reshape(-1, t.shape[-1])
    return None


def load_audio_for_uid(
    uid: str,
    split: str,
    audio_root: Path,
    audio_dim: int,
    cache: dict[tuple[str, str], torch.Tensor],
) -> torch.Tensor:
    key = (split, uid)
    if key in cache:
        return cache[key]

    default_audio = torch.zeros(1, audio_dim, dtype=torch.float32)
    audio_file = audio_root / split / f"{uid}.pt"

    if not audio_file.is_file():
        cache[key] = default_audio
        return default_audio

    obj = torch.load(audio_file, map_location="cpu", weights_only=False)

    candidate: torch.Tensor | None = None
    if torch.is_tensor(obj):
        candidate = _to_2d_float_tensor(obj)
    elif isinstance(obj, dict):
        # Prefer explicit audio keys first.
        for name in ("audio", "embeddings", "features", "x"):
            if name in obj:
                candidate = _to_2d_float_tensor(obj[name])
                if candidate is not None:
                    break

        # Fallback: first tensor-like value that can be interpreted as 2D features.
        if candidate is None:
            for value in obj.values():
                candidate = _to_2d_float_tensor(value)
                if candidate is not None:
                    break

    if candidate is None or candidate.numel() == 0:
        candidate = default_audio

    cache[key] = candidate
    return candidate


def build_dataset(
    source_path: str,
    split_file: str,
    gt_path: str,
    json_path: str,
    clip_frames: int,
    img_size: int,
    stride: int,
    min_seg_frames: int,
    use_track: bool,
) -> Dataset:
    # Use mode='val' for deterministic transforms during export.
    return ViTImagerLoader(
        source_path=source_path,
        split_file=split_file,
        gt_path=gt_path,
        json_path=json_path,
        stride=stride,
        clip_frames=clip_frames,
        img_size=img_size,
        min_seg_frames=min_seg_frames,
        mode="val",
        use_track=use_track,
    )


@torch.no_grad()
def export_split(
    model: torch.nn.Module,
    dataset: ViTImagerLoader,
    split: str,
    output_root: Path,
    audio_root: Path,
    audio_dim: int,
    device: torch.device,
) -> None:
    out_dir = output_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    audio_cache: dict[tuple[str, str], torch.Tensor] = {}

    for idx in range(len(dataset)):
        meta = dataset.samples[idx]
        uid = str(meta.get("uid", f"unknown_{idx}"))
        personid = int(meta.get("personid", 0))

        item = dataset[idx]
        if isinstance(item, tuple) and len(item) == 3:
            clip, track, label = item
            clip = clip.unsqueeze(0).to(device)
            track = track.unsqueeze(0).to(device)
            feats = model.extract_tokens(clip, track)
        else:
            clip, label = item
            clip = clip.unsqueeze(0).to(device)
            feats = model.extract_tokens(clip)

        frame_tokens = feats["frame_tokens"].squeeze(0).detach().cpu().float()  # [T, D]
        t_v = int(frame_tokens.shape[0])
        labels = torch.full((t_v,), float(int(label)), dtype=torch.float32)

        audio = load_audio_for_uid(
            uid=uid,
            split=split,
            audio_root=audio_root,
            audio_dim=audio_dim,
            cache=audio_cache,
        )

        out_path = out_dir / f"{uid}_p{personid}_{idx:06d}.pt"
        torch.save(
            {
                "visual": frame_tokens,
                "audio": audio,
                "labels": labels,
                "uid": uid,
                "personid": personid,
            },
            out_path,
        )
        saved += 1

        if (idx + 1) % 200 == 0:
            print(f"[Export:{split}] {idx + 1}/{len(dataset)}")

    print(f"[Export:{split}] done: {saved} files -> {out_dir}")


def main() -> None:
    args = parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[Device] {device}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    audio_root = Path(args.audio_embed_root)

    with open(output_root / "emb2_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    model = build_visual_model(args, device)

    train_ds = build_dataset(
        source_path=args.source_path,
        split_file=args.train_file,
        gt_path=args.gt_path,
        json_path=args.json_path,
        clip_frames=args.clip_frames,
        img_size=args.img_size,
        stride=args.stride,
        min_seg_frames=args.min_seg_frames,
        use_track=args.use_track,
    )
    val_ds = build_dataset(
        source_path=args.source_path,
        split_file=args.val_file,
        gt_path=args.gt_path,
        json_path=args.json_path,
        clip_frames=args.clip_frames,
        img_size=args.img_size,
        stride=args.stride,
        min_seg_frames=args.min_seg_frames,
        use_track=args.use_track,
    )

    export_split(model, train_ds, "train", output_root, audio_root, args.audio_dim, device)
    export_split(model, val_ds, "val", output_root, audio_root, args.audio_dim, device)

    print("[Done] Fusion-ready files exported.")


if __name__ == "__main__":
    main()
