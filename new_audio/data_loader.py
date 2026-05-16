"""
Audio Pipeline - Data Loader
==============================
Mirrors the visual pipeline structure exactly:
    - Same context window approach
  - Same extracted annotation format
  - Aligns audio segments to video frames via fps/sample_rate

Audio alignment:
  30fps video, 16kHz audio, HuBERT outputs 1 frame per 20ms (50Hz)
  1 video frame = 16000/30 = 533 audio samples
  context_before=8 video frames = 8/30 = 0.267s = 4267 audio samples
"""

from __future__ import annotations
from transformers import WhisperFeatureExtractor
import os
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import soundfile as sf
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────

@dataclass
class AudioConfig:
    # paths
    wave_path          : str   = "/DATA/G17/Data/wave/"
    train_annotations  : str   = ""
    val_annotations    : str   = ""

    # audio params
    sample_rate        : int   = 16000
    fps                : int   = 30

    # clip window (in VIDEO frames — automatically converted to audio samples)
    context_before_frames: int = 8     # video frames before center
    context_after_frames : int = 8     # video frames after center
    clip_video_frames    : int = 16    # center clip length in video frames

    # sampling
    pos_stride         : int   = 4     # same as visual pipeline
    neg_stride         : int   = 60    # same as visual pipeline
    val_pos_stride     : int   = 4
    val_neg_stride     : int   = 60

    # model
    output_dim         : int   = 768
    freeze_backbone    : bool  = True

    # loader
    batch_size         : int   = 16
    num_workers        : int   = 8
    pin_memory         : bool  = True

    @property
    def total_video_frames(self) -> int:
        return self.context_before_frames + self.clip_video_frames + self.context_after_frames

    @property
    def total_audio_samples(self) -> int:
        """Total audio samples for the full context window."""
        return int(self.total_video_frames / self.fps * self.sample_rate)

    @property
    def samples_per_video_frame(self) -> int:
        return int(self.sample_rate / self.fps)


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _resolve_wave_path(wave_dir: str, clip_uid: str) -> str | None:
    """Try common wav file locations."""
    for fmt in (
        os.path.join(wave_dir, f"{clip_uid}.wav"),
        os.path.join(wave_dir, clip_uid, "audio.wav"),
        os.path.join(wave_dir, clip_uid, "aud.wav"),
    ):
        if os.path.isfile(fmt):
            return fmt
    return None


def _load_audio_segment(
    wav_path       : str,
    center_frame   : int,
    before_frames  : int,
    after_frames   : int,
    clip_frames    : int,
    fps            : int   = 30,
    sample_rate    : int   = 16000,
) -> torch.Tensor:
    """
    Load audio segment centered on center_frame.
    Returns [num_samples] float32 tensor, zero-padded if needed.
    """
    total_frames  = before_frames + clip_frames + after_frames
    total_samples = int(total_frames / fps * sample_rate)
    start_frame   = center_frame - before_frames
    start_sample  = max(0, int(start_frame / fps * sample_rate))

    try:
        track = sf.SoundFile(wav_path)
        total_wav_samples = track.frames

        # handle out-of-bounds
        if start_sample >= total_wav_samples:
            return torch.zeros(total_samples, dtype=torch.float32)

        read_samples = min(total_samples, total_wav_samples - start_sample)
        track.seek(start_sample)
        audio = track.read(read_samples, dtype='float32')
        track.close()

        # stereo → mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # pad if needed
        if len(audio) < total_samples:
            audio = np.pad(audio, (0, total_samples - len(audio)))

        return torch.tensor(audio[:total_samples], dtype=torch.float32)

    except Exception as e:
        return torch.zeros(total_samples, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
#  Sample builder (mirrors visual pipeline)
# ──────────────────────────────────────────────────────────────

def _build_audio_samples(
    annotations_path    : str,
    wave_path           : str,
    pos_stride          : int = 4,
    neg_stride          : int = 60,
    context_before      : int = 8,
    context_after       : int = 8,
    clip_frames         : int = 16,
) -> list[dict]:
        """
        Build audio samples from frame-wise extracted annotations.
        """
    if not annotations_path or not os.path.isfile(annotations_path):
        return []

    print(f"[AudioLoader] Loading annotations from {annotations_path}")
    records = _load_json(annotations_path)
    if not isinstance(records, list) or not records:
        return []

    # group by (clip_uid, person_id)
    grouped: dict[tuple[str, int], dict[int, dict]] = {}
    wav_cache: dict[str, str | None] = {}

    for rec in records:
        if not isinstance(rec, dict):
            continue
        clip_uid = str(rec.get("clip_uid", "")).strip()
        if not clip_uid:
            continue
        try:
            person_id = int(rec.get("person_id"))
            frame_id  = int(rec.get("frame"))
        except (TypeError, ValueError):
            continue
        if person_id == 0 or frame_id <= 0:
            continue
        try:
            label = 1 if int(rec.get("ttm_label", 0)) == 1 else 0
        except (TypeError, ValueError):
            label = 0

        key = (clip_uid, person_id)
        grouped.setdefault(key, {})[frame_id] = {
            "label"    : label,
            "video_uid": rec.get("video_uid"),
        }

    samples: list[dict] = []

    for (clip_uid, person_id), frame_map in grouped.items():
        if not frame_map:
            continue

        # resolve wav file
        if clip_uid not in wav_cache:
            wav_cache[clip_uid] = _resolve_wave_path(wave_path, clip_uid)
        wav_path_resolved = wav_cache[clip_uid]
        if wav_path_resolved is None:
            continue

        # sampling (positive and negative strides)
        pos_fids = sorted(fid for fid, info in frame_map.items() if info["label"] == 1)
        neg_fids = sorted(fid for fid, info in frame_map.items() if info["label"] == 0)

        sampled_pos = pos_fids[::pos_stride]
        sampled_neg = neg_fids[::neg_stride]
        center_fids = sorted(sampled_pos + sampled_neg)

        for center_fid in center_fids:
            info  = frame_map.get(center_fid)
            if info is None:
                continue
            label = info["label"]

            samples.append({
                "clip_uid"  : clip_uid,
                "person_id" : person_id,
                "center_fid": center_fid,
                "label"     : label,
                "wav_path"  : wav_path_resolved,
            })

    return samples


# ──────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────

class AudioTTMDataset(Dataset):
    """
    Loads audio segments centered on annotated frames.
    Returns (waveform [num_samples], label) for training
    or (waveform, label) for validation.

    Waveform is raw float32 at 16kHz — HuBERT feature extractor
    handles normalization internally.
    """

    def __init__(
        self,
        annotations_path    : str,
        wave_path           : str,
        pos_stride          : int  = 4,
        neg_stride          : int  = 60,
        context_before      : int  = 8,
        context_after       : int  = 8,
        clip_frames         : int  = 16,
        fps                 : int  = 30,
        sample_rate         : int  = 16000,
        mode                : Literal["train", "val"] = "train",
    ):
        self.wave_path      = wave_path
        self.context_before = context_before
        self.context_after  = context_after
        self.clip_frames    = clip_frames
        self.fps            = fps
        self.sample_rate    = sample_rate
        self.training       = (mode == "train")
        self.whisper_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
        print(f"[AudioLoader] Building '{mode}' samples …")
        self.samples = _build_audio_samples(
            annotations_path = annotations_path,
            wave_path        = wave_path,
            pos_stride       = pos_stride,
            neg_stride       = neg_stride,
            context_before   = context_before,
            context_after    = context_after,
            clip_frames      = clip_frames,
        )

        pos = sum(s["label"] == 1 for s in self.samples)
        neg = len(self.samples) - pos
        print(f"[AudioLoader] '{mode}': {len(self.samples)} samples  "
              f"pos={pos}  neg={neg}  ratio=1:{neg/max(pos,1):.1f}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        s = self.samples[idx]

        waveform = _load_audio_segment(
            wav_path      = s["wav_path"],
            center_frame  = s["center_fid"],
            before_frames = self.context_before,
            after_frames  = self.context_after,
            clip_frames   = self.clip_frames,
            fps           = self.fps,
            sample_rate   = self.sample_rate,
        )

        # light augmentation during training
        # light augmentation during training
        # light augmentation during training
        if self.training:
            waveform = self._augment(waveform)

        # FIXED: Tell the processor to pad to the full 3000 frames (max_length)
        mel = self.whisper_processor(
            waveform.numpy(), 
            sampling_rate=self.sample_rate, 
            return_tensors="pt",
            padding="max_length" # <--- THIS ENSURES 3000 FRAMES
        ).input_features.squeeze(0) 

        return mel, s["label"]

    def _augment(self, waveform: Tensor) -> Tensor:
        """Light augmentations for audio — mirrors visual pipeline approach."""
        # gaussian noise
        if random.random() < 0.3:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise

        # random gain
        if random.random() < 0.3:
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain

        # random time shift (up to 0.1s)
        if random.random() < 0.2:
            shift = random.randint(-1600, 1600)  # ±0.1s at 16kHz
            if shift > 0:
                waveform = torch.cat([torch.zeros(shift), waveform[:-shift]])
            elif shift < 0:
                waveform = torch.cat([waveform[-shift:], torch.zeros(-shift)])

        return waveform.clamp(-1.0, 1.0)


# ──────────────────────────────────────────────────────────────
#  Collate — pads waveforms to same length in batch
# ──────────────────────────────────────────────────────────────

def audio_collate_fn(batch: list[tuple[Tensor, int]]) -> tuple[Tensor, Tensor]:
    """Since all mels are now exactly 3000 frames, we just stack them."""
    mels, labels = zip(*batch)
    return torch.stack(mels), torch.tensor(labels, dtype=torch.long)

# ──────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────

def get_audio_loader(
    cfg : AudioConfig,
    mode: Literal["train", "val"],
) -> DataLoader:

    annotations = cfg.train_annotations if mode == "train" else cfg.val_annotations
    pos_stride  = cfg.pos_stride        if mode == "train" else cfg.val_pos_stride
    neg_stride  = cfg.neg_stride        if mode == "train" else cfg.val_neg_stride

    dataset = AudioTTMDataset(
        annotations_path = annotations,
        wave_path        = cfg.wave_path,
        pos_stride       = pos_stride,
        neg_stride       = neg_stride,
        context_before   = cfg.context_before_frames,
        context_after    = cfg.context_after_frames,
        clip_frames      = cfg.clip_video_frames,
        fps              = cfg.fps,
        sample_rate      = cfg.sample_rate,
        mode             = mode,
    )

    return DataLoader(
        dataset,
        batch_size         = cfg.batch_size,
        shuffle            = (mode == "train"),
        num_workers        = cfg.num_workers,
        pin_memory         = cfg.pin_memory,
        collate_fn         = audio_collate_fn,
        persistent_workers = (cfg.num_workers > 0),
        prefetch_factor    = 4 if cfg.num_workers > 0 else None,
        drop_last          = (mode == "train"),
    )