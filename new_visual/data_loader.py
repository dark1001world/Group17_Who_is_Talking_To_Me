"""
Visual Pipeline - Data Loader (Final Corrected Version)
=========================================================
Fixes applied:
  1. Temporally consistent transforms — same crop/flip/jitter for all frames
  2. Light augmentations suitable for face crops
  3. Skip zero-length and short segments (< 8 frames)
    4. Correct label extraction — tag/tags value 1 = positive
  5. img_00001.jpg frame naming format
  6. Dense clip sampling for Video Swin
"""

from __future__ import annotations

import os
import json
import random
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import cv2
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)



#  Config


@dataclass
class ViTVisualConfig:
    source_path  : str   = "data/video_imgs"
    json_path    : str   = "data/json_original"
    gt_path      : str   = "data/result_TTM"
    train_file   : str   = "data/split/train.list"
    val_file     : str   = "data/split/val.list"
    train_annotations: str = ""
    val_annotations  : str = ""
    extracted_min_seg_frames: int = 8
    extracted_training_mode: Literal["segment", "context_clip"] = "segment"
    context_before_frames: int = 8
    context_after_frames : int = 8
    test_path    : str   = ""
    clip_frames  : int   = 16
    img_size     : int   = 128
    train_stride : int   = 4
    val_stride   : int   = 4
    test_stride  : int   = 1
    batch_size   : int   = 6
    num_workers  : int   = 10
    pin_memory   : bool  = True
    min_seg_frames: int  = 8      # skip segments shorter than this
    use_track    : bool  = False



#  Normalization constants


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# Per-process caches. Each DataLoader worker gets its own copy, which still
# removes the repeated JSON parsing cost within that worker.
_FACE_BBOX_CACHE: dict[tuple[str, str], dict[str, tuple[int, int, int, int]]] = {}
_TRACK_FRAME_CACHE: dict[tuple[str, str], dict[int, dict]] = {}



#  Val / Test transform  (deterministic)


def get_val_transform(img_size: int = 128) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(int(img_size * 256 / 224), antialias=True),
        T.CenterCrop(img_size),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────
#  Temporally consistent train augmentation
#  Same spatial transform applied to EVERY frame in a clip
# ──────────────────────────────────────────────────────────────

class ConsistentClipTransform:
    """
    Generates augmentation parameters ONCE per clip,
    applies identical spatial transform to every frame.
    Uses only PIL + stable torchvision.transforms.functional API.
    """

    def __init__(self, img_size: int = 128):
        self.img_size  = img_size
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, frames: list[Image.Image]) -> list[Tensor]:
        # sample shared params ONCE for entire clip
        i, j, h, w = T.RandomResizedCrop.get_params(
            frames[0],
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
        )
        do_flip    = random.random() < 0.5
        brightness = random.uniform(0.8, 1.2)
        contrast   = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.9, 1.1)

        processed = []
        for frame in frames:
            # 1. same spatial crop — PIL → PIL
            frame = TF.resized_crop(
                frame, i, j, h, w,
                [self.img_size, self.img_size],
            )
            # 2. same flip — PIL → PIL
            if do_flip:
                frame = TF.hflip(frame)

            # 3. PIL → Tensor [C, H, W] float32 in [0, 1]
            frame = TF.to_tensor(frame)

            # 4. same color jitter — Tensor → Tensor
            frame = TF.adjust_brightness(frame, brightness)
            frame = TF.adjust_contrast(frame, contrast)
            frame = TF.adjust_saturation(frame, saturation)

            # 5. normalize
            frame = self.normalize(frame)

            processed.append(frame)

        return processed


#  Helpers


def _load_uid_list(path: str) -> list[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _load_frame(p: str) -> Image.Image:
    img_bgr = cv2.imread(p)
    if img_bgr is None:
        return Image.new("RGB", (1920, 1080), (0, 0, 0))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def _load_face_bboxes(uid: str, json_root: str) -> dict[str, tuple[int, int, int, int]]:
    """Load per-frame face boxes keyed by '<frame>:<personid>' for one UID."""
    cache_key = (json_root, uid)
    cached = _FACE_BBOX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    uid_dir = os.path.join(json_root, uid)
    if not os.path.isdir(uid_dir):
        _FACE_BBOX_CACHE[cache_key] = {}
        return _FACE_BBOX_CACHE[cache_key]

    out: dict[str, tuple[int, int, int, int]] = {}
    for track_file in glob.glob(os.path.join(uid_dir, "*.json")):
        try:
            track = _load_json(track_file)
        except Exception:
            continue
        if not isinstance(track, list):
            continue

        for frame in track:
            try:
                pid_raw = frame.get("Person ID", "")
                pid = str(pid_raw).replace("'", "")
                if not pid:
                    continue

                fid = int(frame["frameNumber"])
                x = float(frame["x"])
                y = float(frame["y"])
                w = float(frame["width"])
                h = float(frame["height"])
                if w <= 0 or h <= 0 or fid <= 0:
                    continue

                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = max(x1 + 1, int(x + w))
                y2 = max(y1 + 1, int(y + h))
                out[f"{fid}:{pid}"] = (x1, y1, x2, y2)
            except (KeyError, TypeError, ValueError):
                continue

    _FACE_BBOX_CACHE[cache_key] = out
    return out


def _crop_face_or_black(img: Image.Image, bbox: tuple[int, int, int, int] | None) -> Image.Image:
    """Crop face with bbox; if missing/invalid, return a black frame of same size."""
    if bbox is None:
        return Image.new("RGB", img.size, (0, 0, 0))

    x1, y1, x2, y2 = bbox
    w, h = img.size
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", img.size, (0, 0, 0))
    return img.crop((x1, y1, x2, y2))


def _resolve_frame_path(frame_dir: str, fid: int) -> str | None:
    """Try all known naming formats."""
    for fmt in (
        f"img_{fid:05d}.jpg",
        f"img_{fid:06d}.jpg",
        f"{fid:06d}.jpg",
        f"{fid:05d}.jpg",
        f"{fid}.jpg",
        f"img_{fid:05d}.png",
        f"{fid:06d}.png",
    ):
        p = os.path.join(frame_dir, fmt)
        if os.path.isfile(p):
            return p
    return None


def _load_track_frame_map(uid: str, json_path: str) -> dict[str, dict]:
    """Load (and cache) '<frame>:<personid>' -> raw track entry map for one UID."""
    cache_key = (json_path, uid)
    cached = _TRACK_FRAME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    track_dir = os.path.join(json_path, uid)
    if not os.path.isdir(track_dir):
        _TRACK_FRAME_CACHE[cache_key] = {}
        return _TRACK_FRAME_CACHE[cache_key]

    frame_bbox: dict[str, dict] = {}
    for track_file in os.listdir(track_dir):
        if not track_file.endswith('.json'):
            continue
        try:
            entries = _load_json(os.path.join(track_dir, track_file))
            for e in entries:
                fid = e.get('frameNumber')
                pid_raw = e.get('Person ID', "")
                pid = str(pid_raw).replace("'", "")
                if fid is None or not pid:
                    continue
                key = f"{int(fid)}:{pid}"
                if key not in frame_bbox:
                    frame_bbox[key] = e
        except Exception:
            continue

    _TRACK_FRAME_CACHE[cache_key] = frame_bbox
    return frame_bbox

def _load_track_features(
    uid      : str,
    json_path: str,
    personid : int,
    fids     : list[int],
    frame_w  : int = 1920,
    frame_h  : int = 1080,
    frame_bbox: dict[str, dict] | None = None,
) -> "torch.Tensor":
    """
    Load per-frame spatial features from track JSONs in json_original/.
 
    Features per frame (6 total):
      cx, cy : normalized face center (0-1)
      size   : normalized face area   (0-1)
      dx, dy : frame-to-frame movement
      ds     : face size change
 
    Returns [T, 6] float32 tensor.
    Falls back to zeros if track data unavailable.
    """
    import torch
 
    T = len(fids)

    if frame_bbox is None:
        frame_bbox = _load_track_frame_map(uid, json_path)
 
    if not frame_bbox:
        return torch.zeros(T, 6, dtype=torch.float32)
 
    # ── extract features per requested frame ──
    feats = []
    prev_cx, prev_cy, prev_size = 0.5, 0.5, 0.05
 
    pid = str(personid)
    for fid in fids:
        key = f"{int(fid)}:{pid}"
        if key in frame_bbox:
            e  = frame_bbox[key]
            x  = float(e.get('x', prev_cx * frame_w))
            y  = float(e.get('y', prev_cy * frame_h))
            h  = float(e.get('height', 0.0))
            w  = float(e.get('width',  0.0))
            cx = (x + 0.5 * w) / max(frame_w, 1)
            cy = (y + 0.5 * h) / max(frame_h, 1)
            size = (h * w) / (frame_w * frame_h + 1e-6)
            # clamp to valid range
            cx   = max(0.0, min(1.0, cx))
            cy   = max(0.0, min(1.0, cy))
            size = max(0.0, min(1.0, size))
        else:
            # use previous frame values (person still there)
            cx, cy, size = prev_cx, prev_cy, prev_size
 
        dx = cx   - prev_cx
        dy = cy   - prev_cy
        ds = size - prev_size
 
        feats.append([cx, cy, size, dx, dy, ds])
        prev_cx, prev_cy, prev_size = cx, cy, size
 
    return torch.tensor(feats, dtype=torch.float32)   # [T, 6]
 

#  Sample builder


def _build_samples(
    uid           : str,
    gt_path       : str,
    source_path   : str,
    json_path     : str,
    stride        : int,
    min_seg_frames: int = 8,
) -> list[dict]:
    gt_file   = os.path.join(gt_path,    f"{uid}.json")
    frame_dir = os.path.join(source_path, uid)

    if not os.path.isfile(gt_file) or not os.path.isdir(frame_dir):
        return []

    segments = _load_json(gt_file)
    if not isinstance(segments, list) or not segments:
        return []

    # get all available frame numbers from disk
    all_frames = sorted([
        int(f.replace("img_", "").replace(".jpg", "").replace(".png", ""))
        for f in os.listdir(frame_dir)
        if (f.startswith("img_") or f[0].isdigit()) and
           (f.endswith(".jpg") or f.endswith(".png"))
    ])
    if not all_frames:
        return []

    face_bboxes = _load_face_bboxes(uid, json_path)

    frame_set = set(all_frames)
    samples   = []

    for seg in segments:
        # Person ID from annotation (kept for filtering/traceability).
        try:
            personid = int(seg.get("label", 0))
        except (TypeError, ValueError):
            personid = 0

        start = int(seg["start_frame"])
        end   = int(seg["end_frame"])

        # skip zero-length or too-short segments
        if end - start < min_seg_frames:
            continue

        # Binary target: talking-to-me is positive only when tag/tags == 1.
        tag_value = seg.get("tags", seg.get("tag", None))
        try:
            tag_value = int(tag_value) if tag_value is not None else 1
        except (TypeError, ValueError):
            tag_value = 0
        label = 1 if tag_value == 0 else 0

        # Match starter pipeline behavior: ignore unknown/invalid person id.
        if personid == 0:
            continue

        # get frames in segment that exist on disk
        seg_frames = [
            f for f in range(start, end + 1, stride)
            if f in frame_set
        ]

        # fallback: nearest frames in range
        if not seg_frames:
            seg_frames = [
                f for f in all_frames
                if start <= f <= end
            ][::stride]

        if not seg_frames:
            continue

        frame_items = []
        for f in seg_frames:
            p = _resolve_frame_path(frame_dir, f)
            if not p:
                continue
            key = f"{f}:{personid}"
            frame_items.append({"path": p, "bbox": face_bboxes.get(key), "fid": f})

        if len(frame_items) >= 2:
            samples.append({
                "uid"     : uid,
                "personid": personid,
                "frames"  : frame_items,
                "fids"    : [it["fid"] for it in frame_items],
                "label"   : label,
            })

    return samples


def _resolve_frame_dir_from_annotation(source_path: str, clip_uid: str, video_uid: str | None) -> str | None:
    """Resolve frame directory from annotation fields for common repository layouts."""
    candidates = [
        os.path.join(source_path, clip_uid),
    ]
    if video_uid:
        candidates.extend([
            os.path.join(source_path, video_uid, clip_uid),
            os.path.join(source_path, video_uid),
        ])

    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def _parse_bbox(raw_bbox) -> tuple[int, int, int, int] | None:
    """Parse extracted bbox format [x, y, w, h] into (x1, y1, x2, y2)."""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x = float(raw_bbox[0])
        y = float(raw_bbox[1])
        w = float(raw_bbox[2])
        h = float(raw_bbox[3])
    except (TypeError, ValueError):
        return None

    if w <= 0 or h <= 0:
        return None

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = max(x1 + 1, int(x + w))
    y2 = max(y1 + 1, int(y + h))
    return (x1, y1, x2, y2)


def _build_samples_from_frame_annotations(
    annotations_path: str,
    source_path: str,
    stride: int,
    min_seg_frames: int = 8,
) -> list[dict]:
    """
    Build clip-level samples from frame-wise extracted annotations.

    Expected record schema:
      {
        "video_uid": str,
        "clip_uid": str,
        "person_id": int,
        "frame": int,
        "bbox": [x, y, w, h],
        "ttm_label": 0|1,
      }
    """
    if not annotations_path or not os.path.isfile(annotations_path):
        return []

    records = _load_json(annotations_path)
    if not isinstance(records, list) or not records:
        return []

    grouped: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        clip_uid = str(rec.get("clip_uid", "")).strip()
        person_id_raw = rec.get("person_id", None)
        if not clip_uid or person_id_raw is None:
            continue

        try:
            person_id = int(person_id_raw)
            frame_id = int(rec.get("frame"))
        except (TypeError, ValueError):
            continue

        if person_id == 0 or frame_id <= 0:
            continue

        label_raw = rec.get("ttm_label", 0)
        try:
            label = 1 if int(label_raw) == 1 else 0
        except (TypeError, ValueError):
            label = 0

        bbox = _parse_bbox(rec.get("bbox"))
        if bbox is None:
            continue

        group_key = (clip_uid, str(person_id))
        grouped.setdefault(group_key, []).append({
            "video_uid": rec.get("video_uid"),
            "clip_uid": clip_uid,
            "personid": person_id,
            "fid": frame_id,
            "bbox": bbox,
            "label": label,
        })

    samples: list[dict] = []
    for (clip_uid, _person_id_key), rows in grouped.items():
        rows.sort(key=lambda r: r["fid"])

        segments: list[list[dict]] = []
        current: list[dict] = []
        prev_fid: int | None = None
        prev_label: int | None = None

        for row in rows:
            fid = row["fid"]
            label = row["label"]
            if (
                prev_fid is None
                or prev_label is None
                or (fid - prev_fid) > 1
                or label != prev_label
            ):
                if current:
                    segments.append(current)
                current = [row]
            else:
                current.append(row)
            prev_fid = fid
            prev_label = label

        if current:
            segments.append(current)

        for seg in segments:
            if len(seg) < min_seg_frames:
                continue

            video_uid = seg[0].get("video_uid")
            frame_dir = _resolve_frame_dir_from_annotation(source_path, clip_uid, str(video_uid) if video_uid else None)
            if frame_dir is None:
                continue

            seg_rows = seg[::max(stride, 1)]
            frame_items = []
            for row in seg_rows:
                frame_path = _resolve_frame_path(frame_dir, row["fid"])
                if not frame_path:
                    continue
                frame_items.append({
                    "path": frame_path,
                    "bbox": row["bbox"],
                    "fid": row["fid"],
                })

            if len(frame_items) >= 2:
                samples.append({
                    "uid": clip_uid,
                    "personid": seg[0]["personid"],
                    "frames": frame_items,
                    "fids": [it["fid"] for it in frame_items],
                    "label": seg[0]["label"],
                })

    return samples


def _build_context_clips_from_frame_annotations(
    annotations_path: str,
    source_path: str,
    center_stride: int,
    context_before: int,
    context_after: int,
) -> list[dict]:
    """
    Build center-labeled clips with left/right temporal context.

    Each sample label is the center frame label, while frames include both
    preceding and following context for better temporal discrimination.
    """
    if not annotations_path or not os.path.isfile(annotations_path):
        return []

    records = _load_json(annotations_path)
    if not isinstance(records, list) or not records:
        return []

    grouped: dict[tuple[str, int], dict[int, dict]] = {}
    frame_dir_cache: dict[tuple[str, str], str | None] = {}

    for rec in records:
        if not isinstance(rec, dict):
            continue

        clip_uid = str(rec.get("clip_uid", "")).strip()
        if not clip_uid:
            continue

        try:
            person_id = int(rec.get("person_id"))
            fid = int(rec.get("frame"))
        except (TypeError, ValueError):
            continue

        if person_id == 0 or fid <= 0:
            continue

        bbox = _parse_bbox(rec.get("bbox"))
        if bbox is None:
            continue

        try:
            label = 1 if int(rec.get("ttm_label", 0)) == 1 else 0
        except (TypeError, ValueError):
            label = 0

        group_key = (clip_uid, person_id)
        frame_map = grouped.setdefault(group_key, {})
        if fid not in frame_map:
            frame_map[fid] = {
                "video_uid": rec.get("video_uid"),
                "bbox": bbox,
                "label": label,
            }

    before = max(0, int(context_before))
    after = max(0, int(context_after))
    step = max(1, int(center_stride))

    samples: list[dict] = []
    for (clip_uid, person_id), frame_map in grouped.items():
        if not frame_map:
            continue

        any_row = next(iter(frame_map.values()))
        video_uid = any_row.get("video_uid")
        cache_key = (str(video_uid) if video_uid else "", clip_uid)
        if cache_key not in frame_dir_cache:
            frame_dir_cache[cache_key] = _resolve_frame_dir_from_annotation(
                source_path,
                clip_uid,
                str(video_uid) if video_uid else None,
            )
        frame_dir = frame_dir_cache[cache_key]
        if frame_dir is None:
            continue

        pos_fids = [fid for fid in frame_map.keys() if int(frame_map[fid]["label"]) == 1]
        neg_fids = [fid for fid in frame_map.keys() if int(frame_map[fid]["label"]) == 0]

        # Dual-stride sampling
        sampled_pos_fids = sorted(pos_fids)[::4]    # every 4th frame ~0.13s
        sampled_neg_fids = sorted(neg_fids)[::60]   # every 60th frame ~2.0s
        
        center_fids = sorted(sampled_pos_fids + sampled_neg_fids)
        for center_fid in center_fids:
            center_info = frame_map.get(center_fid)
            if center_info is None:
                continue

            frame_items = []
            for fid in range(center_fid - before, center_fid + after + 1):
                info = frame_map.get(fid)
                if info is None:
                    continue
                frame_path = _resolve_frame_path(frame_dir, fid)
                if not frame_path:
                    continue
                frame_items.append({
                    "path": frame_path,
                    "bbox": info["bbox"],
                    "fid": fid,
                })

            if len(frame_items) >= 2:
                samples.append({
                    "uid": clip_uid,
                    "personid": person_id,
                    "frames": frame_items,
                    "fids": [it["fid"] for it in frame_items],
                    "label": int(center_info["label"]),
                })

    return samples



#  Clip sampler


def _sample_clip(
    paths       : list,
    clip_frames : int,
    training    : bool,
) -> list:
    n = len(paths)
    if n == 0:
        raise ValueError("Empty frame list")

    if n < clip_frames:
        # For short clips, pad with the last frame to avoid cyclic temporal jumps.
        # Cyclic repetition (e.g., 0,1,2,0,1,2) creates artificial motion spikes.
        return paths + [paths[-1]] * (clip_frames - n)

    if training:
        # random temporal crop
        start = random.randint(0, n - clip_frames)
        return paths[start : start + clip_frames]
    else:
        # uniform sampling
        idxs = np.linspace(0, n - 1, clip_frames, dtype=int)
        return [paths[i] for i in idxs]


# ──────────────────────────────────────────────────────────────
#  Train/Val Dataset
# ──────────────────────────────────────────────────────────────

class ViTImagerLoader(Dataset):
    """
    Returns clips [C, T, H, W] with temporally consistent augmentation.
    """

    def __init__(
        self,
        source_path   : str,
        split_file    : str = "",
        gt_path       : str = "",
        annotations_path: str = "",
        stride        : int  = 4,
        clip_frames   : int  = 16,
        img_size      : int  = 128,
        min_seg_frames: int  = 8,
        extracted_min_seg_frames: int | None = None,
        extracted_training_mode: Literal["segment", "context_clip"] = "segment",
        context_before_frames: int = 8,
        context_after_frames : int = 8,
        mode          : Literal["train", "val"] = "train",
        json_path     : str  = "",
        use_track     : bool = False,       # ← ADD THIS
    ):
        self.clip_frames = clip_frames
        self.training    = (mode == "train")
        self.img_size    = img_size
        self.use_track = use_track
        self.json_path = json_path
        # transforms
        self.train_transform = ConsistentClipTransform(img_size)
        self.val_transform   = get_val_transform(img_size)

        self.samples: list[dict] = []
        if annotations_path:
            print(f"[ViTLoader] Building '{mode}' from extracted annotations: {annotations_path}")
            if extracted_training_mode == "context_clip":
                self.samples = _build_context_clips_from_frame_annotations(
                    annotations_path=annotations_path,
                    source_path=source_path,
                    center_stride=stride,
                    context_before=context_before_frames,
                    context_after=context_after_frames,
                )
            else:
                self.samples = _build_samples_from_frame_annotations(
                    annotations_path=annotations_path,
                    source_path=source_path,
                    stride=stride,
                    min_seg_frames=(
                        extracted_min_seg_frames
                        if extracted_min_seg_frames is not None
                        else min_seg_frames
                    ),
                )
        else:
            uids = _load_uid_list(split_file)
            print(f"[ViTLoader] Building '{mode}' from {len(uids)} UIDs …")
            for uid in uids:
                self.samples.extend(
                    _build_samples(uid, gt_path, source_path, json_path, stride, min_seg_frames)
                )

        # Preload track maps in parent process before DataLoader workers fork.
        # Workers then reuse these read-mostly structures without re-parsing JSON files.
        self._track_frame_maps: dict[str, dict[str, dict]] = {}
        if self.use_track and self.json_path:
            unique_uids = {s["uid"] for s in self.samples}
            for uid in unique_uids:
                self._track_frame_maps[uid] = _load_track_frame_map(uid, self.json_path)

        pos = sum(s["label"] == 1 for s in self.samples)
        neg = len(self.samples) - pos
        print(f"[ViTLoader] '{mode}': {len(self.samples)} samples  "
              f"pos={pos}  neg={neg}  ratio=1:{neg/max(pos,1):.1f}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        s           = self.samples[idx]
        frame_items = _sample_clip(s["frames"], self.clip_frames, self.training)
        frames      = []
        fids        = []
        frame_w     = 1920
        frame_h     = 1080

        for i, item in enumerate(frame_items):
            if isinstance(item, dict):
                p = item.get("path")
                bbox = item.get("bbox")
                fid = item.get("fid", i)
            else:
                p = item
                bbox = None
                fid = i
            img = _load_frame(p)
            if i == 0:
                frame_w, frame_h = img.size
            frames.append(_crop_face_or_black(img, bbox))
            fids.append(int(fid))

        if self.training:
            processed = self.train_transform(frames)
        else:
            processed = [self.val_transform(f) for f in frames]

        clip = torch.stack(processed, dim=1)   # [C, T, H, W]

        if self.use_track and self.json_path:
            track = _load_track_features(
                s["uid"],
                self.json_path,
                s["personid"],
                fids,
                frame_w=frame_w,
                frame_h=frame_h,
                frame_bbox=self._track_frame_maps.get(s["uid"]),
            )                                  # [T, 6]
            return clip, track, s["label"]
        else:
            return clip, s["label"]



#  Test Dataset


class ViTTestImagerLoader(Dataset):
    def __init__(
        self,
        test_path   : str,
        stride      : int = 1,
        clip_frames : int = 16,
        img_size    : int = 128,
    ):
        self.clip_frames   = clip_frames
        self.val_transform = get_val_transform(img_size)
        self.samples       = self._build(test_path, stride)
        print(f"[ViTTestLoader] {len(self.samples)} test samples")

    def _build(self, test_path: str, stride: int) -> list[dict]:
        samples = []
        for uid in sorted(os.listdir(test_path)):
            face_dir = os.path.join(test_path, uid, "face")
            if not os.path.isdir(face_dir):
                face_dir = os.path.join(test_path, uid)
            if not os.path.isdir(face_dir):
                continue

            fids = sorted([
                int(Path(f).stem.replace("img_", ""))
                for f in os.listdir(face_dir)
                if f.endswith((".jpg", ".png"))
            ])
            fids  = fids[::stride]
            paths = [_resolve_frame_path(face_dir, f) for f in fids]
            paths = [p for p in paths if p]
            if paths:
                samples.append({"uid": uid, "frames": paths, "fids": fids})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        s  = self.samples[idx]
        fp = _sample_clip(s["frames"], self.clip_frames, training=False)
        clip = torch.stack(
            [self.val_transform(_load_frame(p)) for p in fp], dim=1
        )
        return clip, {"uid": s["uid"], "fid2pred": s["fids"]}



#  Infer Dataset


class ViTInferImagerLoader(ViTTestImagerLoader):
    pass



#  MixUp  (batch-level, optional)


def mixup_batch(
    clips      : Tensor,
    labels     : Tensor,
    alpha      : float = 0.2,
    num_classes: int   = 2,
) -> tuple[Tensor, Tensor]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    B   = clips.size(0)
    idx = torch.randperm(B, device=clips.device)

    mixed  = lam * clips + (1 - lam) * clips[idx]
    onehot = torch.zeros(B, num_classes, device=clips.device)
    onehot.scatter_(1, labels.unsqueeze(1), 1)
    soft = lam * onehot + (1 - lam) * onehot[idx]
    return mixed, soft



#  Factory


def get_loader(
    cfg        : ViTVisualConfig,
    mode       : Literal["train", "val", "test", "infer"],
    distributed: bool = False,
) -> DataLoader:

    sampler = None

    if mode in ("train", "val"):
        annotation_path = cfg.train_annotations if mode == "train" else cfg.val_annotations
        dataset = ViTImagerLoader(
            source_path    = cfg.source_path,
            split_file     = cfg.train_file if mode == "train" else cfg.val_file,
            gt_path        = cfg.gt_path,
            annotations_path = annotation_path,
            json_path      = cfg.json_path,
            stride         = cfg.train_stride if mode == "train" else cfg.val_stride,
            clip_frames    = cfg.clip_frames,
            img_size       = cfg.img_size,
            min_seg_frames = cfg.min_seg_frames,
            extracted_min_seg_frames = cfg.extracted_min_seg_frames,
            extracted_training_mode = cfg.extracted_training_mode,
            context_before_frames = cfg.context_before_frames,
            context_after_frames = cfg.context_after_frames,
            mode           = mode,
            use_track      = cfg.use_track,
        )
        if distributed and mode == "train":
            from torch.utils.data import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=True)

    elif mode == "test":
        dataset = ViTTestImagerLoader(
            test_path   = cfg.test_path,
            stride      = cfg.test_stride,
            clip_frames = cfg.clip_frames,
            img_size    = cfg.img_size,
        )
    elif mode == "infer":
        dataset = ViTInferImagerLoader(
            test_path   = cfg.test_path,
            stride      = cfg.test_stride,
            clip_frames = cfg.clip_frames,
            img_size    = cfg.img_size,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    shuffle = (mode == "train") and (sampler is None)

    return DataLoader(
        dataset,
        batch_size         = cfg.batch_size,
        shuffle            = shuffle,
        sampler            = sampler,
        num_workers        = cfg.num_workers,
        pin_memory         = cfg.pin_memory,
        persistent_workers = (cfg.num_workers > 0),
        prefetch_factor    = 4 if cfg.num_workers > 0 else None,
        drop_last          = (mode == "train"),
    )
