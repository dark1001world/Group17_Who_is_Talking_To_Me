"""
extract_embeddings.py
──────────────────────
Extract **timestamped audio embeddings** from the fine-tuned (or pretrained)
WhisperTTM model.

For every clip in the annotation JSON, this script:
  1. Loads the audio in sliding 30-s windows
  2. Runs the Whisper encoder (no grad) to get hidden states
  3. Projects to projection_dim (default 512)
  4. Tags every embedding vector with its absolute timestamp in the original clip
  5. Saves a .pt file per clip_uid

Output format (per clip)
────────────────────────
  {
    "clip_uid"       : str,
    "video_uid"      : str,
    "embeddings"     : Tensor (N_frames_total, projection_dim),  # float32
    "timestamps_sec" : Tensor (N_frames_total,),                 # absolute seconds in clip
    "frame_step_sec" : float,                                    # seconds between frames
    "config"         : dict                                      # extraction config snapshot
  }

Usage
─────
  # After training:
  python extract_embeddings.py --config config.yaml \
      --checkpoint /DATA/G17/outputs/ttm_audio/checkpoints/ckpt_best.pt \
      --split train

  # Without fine-tuning (use raw Whisper encoder):
  python extract_embeddings.py --config config.yaml --split val

  # Single clip (debug):
  python extract_embeddings.py --config config.yaml \
      --checkpoint ckpt_best.pt --clip_uid <uid>
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from ttm_dataset import load_annotation_json
from audio_model import WhisperTTM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract")


# ── constants ─────────────────────────────────────────────────────────────────

WHISPER_SR        = 16_000     # Whisper expects 16 kHz
WHISPER_WIN_SECS  = 30.0       # Whisper processes 30-s chunks
WHISPER_WIN_SAMPLES = int(WHISPER_WIN_SECS * WHISPER_SR)   # 480 000
WHISPER_HOP_FRAMES  = 2        # output frames every 20 ms → 50 Hz
# Whisper encoder downsamples 480000 samples → 1500 frames (320x)
ENC_OUT_PER_SEC   = 1500 / 30  # = 50 encoder frames per second


# ── audio loader ──────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = WHISPER_SR) -> torch.Tensor:
    """Load mono 16 kHz waveform. Returns (T,) float32 tensor."""
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.squeeze(0)  # (T,)


def window_waveform(
    waveform: torch.Tensor,
    window_samples: int = WHISPER_WIN_SAMPLES,
    overlap_samples: int = 0,
) -> List[Dict]:
    """
    Split a long waveform into ≤30 s chunks.
    Returns list of {'chunk': Tensor(window_samples,), 'offset_samples': int}
    """
    T = waveform.shape[0]
    step = window_samples - overlap_samples
    windows = []
    start = 0
    while start < T:
        end   = start + window_samples
        chunk = waveform[start:end]
        # Zero-pad last chunk
        if chunk.shape[0] < window_samples:
            chunk = torch.nn.functional.pad(chunk, (0, window_samples - chunk.shape[0]))
        windows.append({"chunk": chunk, "offset_samples": start})
        if end >= T:
            break
        start += step
    return windows


# ── timestamp generator ───────────────────────────────────────────────────────

def make_timestamps(
    n_enc_frames: int,
    offset_samples: int,
    sr: int = WHISPER_SR,
) -> torch.Tensor:
    """
    Return absolute timestamps (in seconds) for each encoder output frame.
    n_enc_frames : number of frames output by Whisper encoder (1500 for 30 s)
    offset_samples: sample offset of this window in the original audio
    """
    frame_step_sec = WHISPER_WIN_SECS / n_enc_frames          # ≈ 0.020 s
    frame_indices  = torch.arange(n_enc_frames, dtype=torch.float32)
    offset_sec     = offset_samples / sr
    return offset_sec + frame_indices * frame_step_sec


# ── model loader ──────────────────────────────────────────────────────────────

def build_model(cfg: dict, checkpoint_path: Optional[str], device: torch.device) -> WhisperTTM:
    model = WhisperTTM(
        model_name=cfg["model"]["backbone"],
        freeze_encoder_layers=cfg["model"]["freeze_encoder_layers"],
        projection_dim=cfg["model"]["projection_dim"],
        dropout=0.0,            # no dropout at inference
        num_classes=cfg["model"]["num_classes"],
        gradient_checkpointing=False,
    )

    if checkpoint_path:
        logger.info("Loading checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected[:5])
    else:
        logger.info("No checkpoint provided – using raw pretrained Whisper encoder weights.")

    model.float()              # convert ALL weights to fp32 after checkpoint load
    model.eval().to(device)
    return model


# ── per-clip extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_clip_embeddings(
    waveform: torch.Tensor,
    model: WhisperTTM,
    fe: WhisperFeatureExtractor,
    device: torch.device,
    cfg: dict,
    layers_to_avg: Optional[list] = None,
) -> Dict:
    """
    Process a single clip (any length) → stacked embeddings + timestamps.
    Returns dict ready to be torch.save'd.
    """
    # Match input dtype to model weights dynamically
    model_dtype = next(model.parameters()).dtype
    dtype = model_dtype

    overlap_sec     = cfg["data"]["clip_overlap_sec"]
    overlap_samples = int(overlap_sec * WHISPER_SR)

    windows = window_waveform(waveform, WHISPER_WIN_SAMPLES, overlap_samples)

    all_embeds     = []
    all_timestamps = []

    for win in windows:
        chunk   = win["chunk"]                             # (480000,)
        offset  = win["offset_samples"]

        # Whisper feature extraction
        feats = fe(
            chunk.numpy(),
            sampling_rate=WHISPER_SR,
            return_tensors="pt",
        ).input_features.to(device, dtype=dtype)          # (1, 80, 3000)

        # Encode
        emb = model.encode(feats, layers_to_avg=layers_to_avg)  # (1, T_enc, proj_dim)
        emb = emb.squeeze(0).float().cpu()                 # (T_enc, proj_dim)

        n_frames = emb.shape[0]
        ts       = make_timestamps(n_frames, offset, sr=WHISPER_SR)  # (T_enc,)

        # If overlapping windows, trim the overlap from the *start* of all
        # windows except the first, so timestamps don't duplicate
        if all_embeds and overlap_samples > 0:
            trim = int(n_frames * overlap_samples / WHISPER_WIN_SAMPLES)
            emb  = emb[trim:]
            ts   = ts[trim:]

        all_embeds.append(emb)
        all_timestamps.append(ts)

    embeddings  = torch.cat(all_embeds,     dim=0)    # (N_total, proj_dim)
    timestamps  = torch.cat(all_timestamps, dim=0)    # (N_total,)

    return {
        "embeddings":     embeddings,
        "timestamps_sec": timestamps,
        "frame_step_sec": float(WHISPER_WIN_SECS / 1500),
    }


# ── main extraction loop ──────────────────────────────────────────────────────

def run_extraction(
    cfg: dict,
    checkpoint_path: Optional[str],
    split: str,
    target_clip_uid: Optional[str],
    device: torch.device,
):
    # ── Setup ────────────────────────────────────────────────────────────────
    ann_dir   = Path(cfg["data"]["annotations_dir"])
    json_name = cfg["data"]["train_json"] if split == "train" else cfg["data"]["val_json"]
    ann_path  = ann_dir / json_name

    embed_dir = Path(cfg["project"]["embed_dir"]) / split
    embed_dir.mkdir(parents=True, exist_ok=True)

    audio_root = Path(cfg["data"]["root"])
    audio_ext  = cfg["data"]["audio_ext"]

    # Determine layer averaging strategy
    layer_cfg = cfg["extraction"].get("layer", "all")
    if layer_cfg == "all":
        layers_to_avg = [-4, -3, -2, -1]   # last 4 encoder layers
    elif isinstance(layer_cfg, int):
        layers_to_avg = [layer_cfg]
    else:
        layers_to_avg = None                # last layer only

    # ── Load model ───────────────────────────────────────────────────────────
    fe    = WhisperFeatureExtractor.from_pretrained(cfg["model"]["backbone"])
    model = build_model(cfg, checkpoint_path, device)

    # ── Load annotations ─────────────────────────────────────────────────────
    clips = load_annotation_json(str(ann_path))
    if target_clip_uid:
        clips = [c for c in clips if c.get("clip_uid") == target_clip_uid]
        if not clips:
            raise ValueError(f"clip_uid '{target_clip_uid}' not found in {ann_path}")

    logger.info("Extracting embeddings for %d clips (split=%s) …", len(clips), split)

    skipped = 0
    for clip in tqdm(clips, desc=f"Extracting [{split}]"):
        clip_uid  = clip.get("clip_uid") or clip.get("id", "")
        video_uid = clip.get("video_uid", "")

        # Locate audio file
        candidates = [
            audio_root / f"{clip_uid}{audio_ext}",
            audio_root / f"{video_uid}_{clip_uid}{audio_ext}",
            audio_root / clip_uid / f"audio{audio_ext}",
            audio_root / video_uid / f"{clip_uid}{audio_ext}",
        ]
        audio_path = next((p for p in candidates if p.exists()), None)

        if audio_path is None:
            skipped += 1
            continue

        # Output path
        out_path = embed_dir / f"{clip_uid}.pt"
        if out_path.exists():
            continue    # already extracted; remove this line to re-extract

        try:
            waveform = load_audio(str(audio_path))

            result = extract_clip_embeddings(
                waveform, model, fe, device, cfg, layers_to_avg=layers_to_avg
            )

            # Attach metadata
            result["clip_uid"]  = clip_uid
            result["video_uid"] = video_uid
            result["config"]    = {
                "backbone":     cfg["model"]["backbone"],
                "proj_dim":     cfg["model"]["projection_dim"],
                "layer_cfg":    layer_cfg,
                "checkpoint":   checkpoint_path,
                "split":        split,
            }

            torch.save(result, out_path)

        except Exception as e:
            logger.error("Failed on clip %s: %s", clip_uid, e)
            skipped += 1
            continue

    total = len(clips)
    logger.info(
        "Done. Saved %d / %d embeddings to %s  (%d skipped)",
        total - skipped, total, embed_dir, skipped,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract timestamped Whisper embeddings for TTM")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned checkpoint .pt  (omit for raw Whisper)")
    parser.add_argument("--split",      default="train", choices=["train", "val", "both"])
    parser.add_argument("--clip_uid",   default=None,
                        help="Extract only a single clip (for debugging)")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if args.split == "both":
        for sp in ["train", "val"]:
            run_extraction(cfg, args.checkpoint, sp, args.clip_uid, device)
    else:
        run_extraction(cfg, args.checkpoint, args.split, args.clip_uid, device)


if __name__ == "__main__":
    main()