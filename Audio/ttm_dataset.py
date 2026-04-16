"""
ttm_dataset.py
──────────────
PyTorch Dataset for the Ego4D TTM (Talking-To-Me) task.

Actual JSON schema (confirmed from av_train.json):
  {
    "videos": [                          ← top-level list key
      {
        "video_uid": "...",
        "clips": [                       ← nested clips per video
          {
            "clip_uid": "...",
            "video_uid": "...",
            "clip_start_sec": 0,
            "clip_end_sec": 299.96,
            "social_segments_talking": [ ← TTM annotations
              {
                "start_time": 11.67,
                "end_time":   12.69,
                "is_at_me":   True       ← only True = positive TTM label
              }, ...
            ]
          }
        ]
      }
    ]
  }
"""

import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from transformers import WhisperFeatureExtractor

logger = logging.getLogger(__name__)


# ── JSON loader ───────────────────────────────────────────────────────────────

def load_annotation_json(json_path: str) -> List[Dict]:
    """
    Flatten the nested videos→clips structure into a plain list of clip dicts.
    Only returns clips where valid=True (if field exists).
    """
    with open(json_path) as f:
        raw = json.load(f)

    clips_flat = []

    videos = raw.get("videos", [])
    for video in videos:
        video_uid = video.get("video_uid", "")
        for clip in video.get("clips", []):
            if not clip.get("valid", True):
                continue
            clip.setdefault("video_uid", video_uid)
            clips_flat.append(clip)

    logger.info("Loaded %d valid clips from %s", len(clips_flat), json_path)
    return clips_flat


# ── label builder ─────────────────────────────────────────────────────────────

def build_frame_labels(
    clip_duration_sec: float,
    social_segments: List[Dict],
    fps: float = 50.0,
) -> torch.Tensor:
    """
    Build binary frame-level label tensor from social_segments_talking.
    Only segments with is_at_me=True are marked as positive (1).
    Returns: LongTensor (num_frames,)
    """
    num_frames = math.ceil(clip_duration_sec * fps)
    labels = torch.zeros(num_frames, dtype=torch.long)

    for seg in social_segments:
        if not seg.get("is_at_me", False):
            continue
        s = int(seg["start_time"] * fps)
        e = int(seg["end_time"]   * fps)
        labels[s : min(e + 1, num_frames)] = 1

    return labels


# ── dataset ───────────────────────────────────────────────────────────────────

class TTMAudioDataset(Dataset):
    """
    Args
    ----
    annotation_json  : path to av_train.json or av_val.json
    audio_root       : root dir containing per-clip audio files
    feature_extractor: HuggingFace WhisperFeatureExtractor
    sample_rate      : 16000 (Whisper requirement)
    max_audio_sec    : 30 (Whisper window size)
    clip_overlap_sec : overlap between consecutive windows for long clips
    split            : "train" or "val"
    audio_ext        : ".wav" or ".mp3"
    max_clips        : if set, only use first N clips (for smoke-test runs)
    """

    LABEL_FPS = 50.0

    def __init__(
        self,
        annotation_json: str,
        audio_root: str,
        feature_extractor: WhisperFeatureExtractor,
        sample_rate: int = 16_000,
        max_audio_sec: float = 30.0,
        clip_overlap_sec: float = 2.0,
        split: str = "train",
        audio_ext: str = ".wav",
        max_clips: Optional[int] = None,
    ):
        self.audio_root      = Path(audio_root)
        self.fe              = feature_extractor
        self.sr              = sample_rate
        self.max_samples     = int(max_audio_sec * sample_rate)
        self.overlap_samples = int(clip_overlap_sec * sample_rate)
        self.split           = split
        self.audio_ext       = audio_ext

        all_clips = load_annotation_json(annotation_json)

        if max_clips:
            all_clips = all_clips[:max_clips]
            logger.info("[%s] Limiting to %d clips (smoke-test mode)", split, max_clips)

        self.samples = self._build_sample_list(all_clips)
        logger.info("[%s] %d clips → %d windows", split, len(all_clips), len(self.samples))

    # ── audio file finder ────────────────────────────────────────────────────

    def _find_audio_file(self, clip_uid: str, video_uid: str) -> Optional[Path]:
        candidates = [
            self.audio_root / f"{clip_uid}{self.audio_ext}",
            self.audio_root / f"{video_uid}_{clip_uid}{self.audio_ext}",
            self.audio_root / clip_uid / f"audio{self.audio_ext}",
            self.audio_root / video_uid / f"{clip_uid}{self.audio_ext}",
            self.audio_root / clip_uid / f"{clip_uid}{self.audio_ext}",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    # ── sample list builder ──────────────────────────────────────────────────

    def _build_sample_list(self, clips: List[Dict]) -> List[Dict]:
        samples = []
        missing = 0

        for clip in clips:
            clip_uid  = clip.get("clip_uid", "")
            video_uid = clip.get("video_uid", "")

            audio_path = self._find_audio_file(clip_uid, video_uid)
            if audio_path is None:
                missing += 1
                continue

            social_segs = clip.get("social_segments_talking", [])
            clip_start  = float(clip.get("clip_start_sec", 0.0))
            clip_end    = float(clip.get("clip_end_sec",   0.0))
            duration    = clip_end - clip_start if clip_end > clip_start else None

            step           = self.max_samples - self.overlap_samples
            offset_samples = 0
            window_idx     = 0

            while True:
                samples.append({
                    "clip_uid":        clip_uid,
                    "video_uid":       video_uid,
                    "audio_path":      str(audio_path),
                    "offset_samples":  offset_samples,
                    "social_segments": social_segs,
                    "duration_sec":    duration,
                    "window_idx":      window_idx,
                    "clip_start_sec":  clip_start,
                })

                if duration is None:
                    break
                total_samples = int(duration * self.sr)
                if offset_samples + self.max_samples >= total_samples:
                    break
                offset_samples += step
                window_idx     += 1

        if missing:
            logger.warning(
                "Audio not found for %d clips – skipped. "
                "Check audio_root and audio_ext in config.yaml",
                missing
            )
        return samples

    # ── augmentation ─────────────────────────────────────────────────────────

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            waveform = waveform * (0.7 + 0.6 * torch.rand(1).item())
        if torch.rand(1).item() < 0.3:
            waveform = waveform + torch.randn_like(waveform) * 0.005
        if torch.rand(1).item() < 0.3:
            T_len = waveform.shape[-1]
            mlen  = int(T_len * 0.05)
            start = torch.randint(0, max(1, T_len - mlen), (1,)).item()
            waveform[..., start : start + mlen] = 0.0
        return waveform.clamp(-1.0, 1.0)

    # ── __getitem__ ──────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        meta = self.samples[idx]

        waveform, orig_sr = torchaudio.load(
            meta["audio_path"],
            frame_offset=meta["offset_samples"],
            num_frames=self.max_samples,
        )

        if orig_sr != self.sr:
            waveform = T.Resample(orig_sr, self.sr)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        T_len = waveform.shape[-1]
        if T_len < self.max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - T_len))
        else:
            waveform = waveform[..., :self.max_samples]

        if self.split == "train":
            waveform = self._augment(waveform)

        input_features = self.fe(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.sr,
            return_tensors="pt",
        ).input_features.squeeze(0)                         # (80, 3000)

        window_start_sec = meta["offset_samples"] / self.sr
        window_dur_sec   = self.max_samples / self.sr

        local_segs = []
        for seg in meta["social_segments"]:
            if not seg.get("is_at_me", False):
                continue
            s = seg["start_time"] - window_start_sec
            e = seg["end_time"]   - window_start_sec
            if e > 0 and s < window_dur_sec:
                local_segs.append({
                    "start_time": max(s, 0.0),
                    "end_time":   min(e, window_dur_sec),
                    "is_at_me":   True,
                })

        frame_labels = build_frame_labels(window_dur_sec, local_segs, self.LABEL_FPS)
        clip_label   = torch.tensor(1 if frame_labels.sum() > 0 else 0, dtype=torch.long)

        return {
            "input_features":   input_features,
            "frame_labels":     frame_labels,
            "clip_label":       clip_label,
            "clip_uid":         meta["clip_uid"],
            "window_idx":       meta["window_idx"],
            "window_start_sec": window_start_sec,
        }


# ── collate ───────────────────────────────────────────────────────────────────

def ttm_collate_fn(batch: List[Dict]) -> Dict:
    max_frames = max(b["frame_labels"].shape[0] for b in batch)
    return {
        "input_features": torch.stack([b["input_features"] for b in batch]),
        "clip_labels":    torch.stack([b["clip_label"]     for b in batch]),
        "frame_labels":   torch.stack([
            torch.nn.functional.pad(
                b["frame_labels"],
                (0, max_frames - b["frame_labels"].shape[0]),
                value=-1
            ) for b in batch
        ]),
        "clip_uids":         [b["clip_uid"]         for b in batch],
        "window_indices":    [b["window_idx"]        for b in batch],
        "window_start_secs": [b["window_start_sec"]  for b in batch],
    }