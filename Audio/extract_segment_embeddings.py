# # """
# # extract_segment_embeddings.py
# # ──────────────────────────────
# # Extracts per-segment audio embeddings from result_ttm JSON files.

# # Logic (confirmed from starter_code/dataset/data_loader.py):
# #     - gt['label']   = person_id  (NOT the TTM label)
# #     - tag/tags == 1 = TTM=1 (talking to me)
# #     - missing tag/tags or value 0 = TTM=0 (not talking to me)
# #   - Audio cut at: onset  = start_frame / 30 * sample_rate
# #                   offset = end_frame   / 30 * sample_rate

# # Output: one .pt file per clip_uid containing ALL segments of that clip
# #   {
# #     "uid"       : str,
# #     "segments"  : [
# #         {
# #           "person_id"   : int,
# #           "label"       : int (0 or 1),
# #           "start_frame" : int,
# #           "end_frame"   : int,
# #           "embedding"   : Tensor (N, 512)
# #         }, ...
# #     ]
# #   }

# # Usage:
# #   python extract_segment_embeddings.py \
# #       --ttm_dir   /DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/result_ttm \
# #       --wave_dir  /DATA/G17/Data/wave \
# #       --output    /DATA/G17/outputs/ttm_segment_embeddings \
# #       --checkpoint /DATA/G17/outputs/ttm_audio/checkpoints/ckpt_best.pt \
# #       --split     train
# # """

# # import os
# # import json
# # import logging
# # import argparse
# # import warnings
# # from pathlib import Path
# # from typing import List, Dict, Optional

# # import torch
# # import torchaudio
# # import torchaudio.transforms as T
# # import numpy as np
# # from tqdm import tqdm
# # from transformers import WhisperFeatureExtractor

# # warnings.filterwarnings("ignore")

# # from audio_model import WhisperTTM

# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s %(levelname)s – %(message)s",
# #     datefmt="%H:%M:%S",
# # )
# # logger = logging.getLogger("seg_extract")

# # # ── Constants ─────────────────────────────────────────────────────
# # VIDEO_FPS        = 30
# # WHISPER_SR       = 16_000
# # WHISPER_WIN_SEC  = 30.0
# # WHISPER_WIN_SAMP = int(WHISPER_WIN_SEC * WHISPER_SR)   # 480000


# # # ── Model loader ──────────────────────────────────────────────────
# # def build_model(checkpoint_path: Optional[str], device: torch.device) -> WhisperTTM:
# #     model = WhisperTTM(
# #         model_name="openai/whisper-large-v3",
# #         freeze_encoder_layers=28,
# #         projection_dim=512,
# #         dropout=0.0,
# #         num_classes=2,
# #         gradient_checkpointing=False,
# #     )

# #     if checkpoint_path and Path(checkpoint_path).exists():
# #         logger.info("Loading checkpoint: %s", checkpoint_path)
# #         ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
# #         state = ckpt.get("model", ckpt)
# #         model.load_state_dict(state, strict=False)
# #     else:
# #         logger.info("No checkpoint – using raw Whisper weights")

# #     model.float()           # convert all weights to fp32
# #     model.eval().to(device)
# #     return model


# # # ── Audio loader ──────────────────────────────────────────────────
# # def load_wav(path: str, target_sr: int = WHISPER_SR) -> torch.Tensor:
# #     """Load mono 16kHz waveform. Returns (T,) float32."""
# #     wav, sr = torchaudio.load(path)
# #     if sr != target_sr:
# #         wav = T.Resample(sr, target_sr)(wav)
# #     if wav.shape[0] > 1:
# #         wav = wav.mean(0, keepdim=True)
# #     return wav.squeeze(0)


# # # ── Per-segment embedding ─────────────────────────────────────────
# # @torch.no_grad()
# # def embed_segment(
# #     waveform_segment: torch.Tensor,   # (T_seg,) already cut
# #     model: WhisperTTM,
# #     fe: WhisperFeatureExtractor,
# #     device: torch.device,
# # ) -> torch.Tensor:
# #     """
# #     Returns embedding for one audio segment.
# #     Handles segments longer than 30s via windowing.
# #     Output: (N_frames, 512)
# #     """
# #     T_seg = waveform_segment.shape[0]

# #     if T_seg == 0:
# #         return torch.zeros(1, 512)

# #     all_embs = []

# #     # Slide 30s windows over segment
# #     start = 0
# #     while start < T_seg:
# #         chunk = waveform_segment[start : start + WHISPER_WIN_SAMP]

# #         # Zero-pad last chunk to exactly 30s
# #         if chunk.shape[0] < WHISPER_WIN_SAMP:
# #             chunk = torch.nn.functional.pad(
# #                 chunk, (0, WHISPER_WIN_SAMP - chunk.shape[0])
# #             )

# #         feats = fe(
# #             chunk.numpy(),
# #             sampling_rate=WHISPER_SR,
# #             return_tensors="pt",
# #         ).input_features.to(device, dtype=torch.float32)   # (1, 80, 3000)

# #         emb = model.encode(feats)          # (1, 1500, 512)
# #         emb = emb.squeeze(0).cpu()         # (1500, 512)

# #         # Only keep frames that correspond to actual audio (not padding)
# #         actual_samples = min(WHISPER_WIN_SAMP, T_seg - start)
# #         actual_frames  = int(1500 * actual_samples / WHISPER_WIN_SAMP)
# #         emb = emb[:actual_frames]

# #         all_embs.append(emb)
# #         start += WHISPER_WIN_SAMP

# #         if start >= T_seg:
# #             break

# #     return torch.cat(all_embs, dim=0)     # (N_frames, 512)


# # # ── Main extraction ───────────────────────────────────────────────
# # def extract(args):
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     logger.info("Device: %s", device)

# #     # Dirs
# #     ttm_dir  = Path(args.ttm_dir)
# #     wave_dir = Path(args.wave_dir)
# #     out_dir  = Path(args.output) / args.split
# #     out_dir.mkdir(parents=True, exist_ok=True)

# #     # Load model + feature extractor
# #     fe    = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
# #     model = build_model(args.checkpoint, device)

# #     # Get clip list for this split
# #     if args.split in ("train", "val"):
# #         split_file = Path(args.starter_code) / "data/split" / f"{args.split}.list"
# #         with open(split_file) as f:
# #             clip_uids = [l.strip() for l in f.readlines()]
# #     else:
# #         # Use all files in ttm_dir
# #         clip_uids = [f.stem for f in ttm_dir.glob("*.json")]

# #     logger.info("Processing %d clips for split=%s", len(clip_uids), args.split)

# #     skipped  = 0
# #     saved    = 0
# #     seg_pos  = 0
# #     seg_neg  = 0

# #     for uid in tqdm(clip_uids, desc=f"[{args.split}]"):

# #         out_path = out_dir / f"{uid}.pt"
# #         if out_path.exists():
# #             continue

# #         # Load TTM segment annotations
# #         ttm_path = ttm_dir / f"{uid}.json"
# #         if not ttm_path.exists():
# #             skipped += 1
# #             continue

# #         with open(ttm_path) as f:
# #             ttm_data = json.load(f)

# #         # Load full clip waveform
# #         wav_path = wave_dir / f"{uid}.wav"
# #         if not wav_path.exists():
# #             skipped += 1
# #             continue

# #         try:
# #             waveform = load_wav(str(wav_path))   # (T_full,)
# #         except Exception as e:
# #             logger.error("Failed loading audio %s: %s", uid, e)
# #             skipped += 1
# #             continue

# #         total_samples = waveform.shape[0]
# #         segments_out  = []

# #         for seg in ttm_data:
# #             person_id   = int(seg["label"])
# #             start_frame = int(seg["start_frame"])
# #             end_frame   = int(seg["end_frame"])
# #             seg_length  = end_frame - start_frame + 1

# #             # TTM label: positive only when tag/tags explicitly equals 1.
# #             tag_value = seg.get("tags", seg.get("tag", 0))
# #             try:
# #                 tag_value = int(tag_value)
# #             except (TypeError, ValueError):
# #                 tag_value = 0
# #             ttm_label = 1 if tag_value == 1 else 0

# #             # Skip person_id=0 (camera wearer) and very short segments
# #             if person_id == 0:
# #                 continue
# #             if seg_length < 3:   # < 3 frames = ~0.1s, too short
# #                 continue

# #             # Convert frames → audio samples (30 fps)
# #             onset  = int(start_frame / VIDEO_FPS * WHISPER_SR)
# #             offset = int(end_frame   / VIDEO_FPS * WHISPER_SR)

# #             # Clamp to actual audio length
# #             onset  = max(0, min(onset,  total_samples))
# #             offset = max(0, min(offset, total_samples))

# #             if offset <= onset:
# #                 continue

# #             # Cut audio segment
# #             seg_wav = waveform[onset:offset]   # (T_seg,)

# #             # Get embedding
# #             try:
# #                 emb = embed_segment(seg_wav, model, fe, device)  # (N, 512)
# #             except Exception as e:
# #                 logger.error("Embed failed for %s seg %d-%d: %s",
# #                              uid, start_frame, end_frame, e)
# #                 continue

# #             segments_out.append({
# #                 "person_id":   person_id,
# #                 "label":       ttm_label,
# #                 "start_frame": start_frame,
# #                 "end_frame":   end_frame,
# #                 "duration_sec": (end_frame - start_frame) / VIDEO_FPS,
# #                 "embedding":   emb,          # (N, 512)
# #             })

# #             if ttm_label == 1:
# #                 seg_pos += 1
# #             else:
# #                 seg_neg += 1

# #         if not segments_out:
# #             skipped += 1
# #             continue

# #         torch.save({"uid": uid, "segments": segments_out}, out_path)
# #         saved += 1

# #     logger.info(
# #         "Done. Saved=%d  Skipped=%d  Pos_segs=%d  Neg_segs=%d  "
# #         "Pos_ratio=%.1f%%",
# #         saved, skipped, seg_pos, seg_neg,
# #         100 * seg_pos / max(seg_pos + seg_neg, 1),
# #     )


# # # ── CLI ───────────────────────────────────────────────────────────
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--ttm_dir",
# #         default="/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/result_ttm")
# #     parser.add_argument("--wave_dir",
# #         default="/DATA/G17/Data/wave")
# #     parser.add_argument("--output",
# #         default="/DATA/G17/outputs/ttm_segment_embeddings")
# #     parser.add_argument("--checkpoint",
# #         default="/DATA/G17/outputs/ttm_audio/checkpoints/ckpt_best.pt")
# #     parser.add_argument("--starter_code",
# #         default="/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code")
# #     parser.add_argument("--split",
# #         default="train", choices=["train", "val", "all"])
# #     parser.add_argument("--device", default="cuda")
# #     args = parser.parse_args()
# #     extract(args)
    

# """
# extract_segment_embeddings.py
# ──────────────────────────────
# Extracts per-segment audio embeddings from result_TTM JSON files.
# Saves ONE JSON file per segment.

# Label logic (from starter_code/dataset/data_loader.py):
#   tags present AND tags==1 → label=1 (talking to me)
#   tags absent  OR  tags==0 → label=0 (not talking to me)
#   gt['label'] = person_id  (NOT TTM label)

# Output filename: {uid}_{person_id}_{start_frame}_{end_frame}.json
# Output format:
# {
#   "uid":         "fc2b2014-...",
#   "person_id":   3,
#   "label":       1,
#   "tag":         1,
#   "start_frame": 4674,
#   "end_frame":   4703,
#   "duration_sec": 0.967,
#   "embedding":   [0.012, -0.034, ...]   ← 512 floats
# }

# Usage:
#   python extract_segment_embeddings.py --split train
#   python extract_segment_embeddings.py --split val
# """

# import os
# import json
# import logging
# import argparse
# import warnings
# from pathlib import Path
# from typing import Optional

# import torch
# import torchaudio
# import torchaudio.transforms as T
# from tqdm import tqdm
# from transformers import WhisperFeatureExtractor

# warnings.filterwarnings("ignore")
# from audio_model import WhisperTTM

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s – %(message)s",
#     datefmt="%H:%M:%S",
# )
# logger = logging.getLogger("seg_extract")

# VIDEO_FPS        = 30
# WHISPER_SR       = 16_000
# WHISPER_WIN_SAMP = int(30.0 * WHISPER_SR)   # 480000


# # ── Model ─────────────────────────────────────────────────────────
# def build_model(checkpoint_path: Optional[str], device: torch.device) -> WhisperTTM:
#     model = WhisperTTM(
#         model_name="openai/whisper-large-v3",
#         freeze_encoder_layers=28,
#         projection_dim=512,
#         dropout=0.0,
#         num_classes=2,
#         gradient_checkpointing=False,
#     )
#     if checkpoint_path and Path(checkpoint_path).exists():
#         logger.info("Loading checkpoint: %s", checkpoint_path)
#         ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
#         state = ckpt.get("model", ckpt)
#         model.load_state_dict(state, strict=False)
#     else:
#         logger.info("No checkpoint – using raw Whisper weights")
#     model.float().eval().to(device)
#     return model


# # ── Audio ─────────────────────────────────────────────────────────
# def load_wav(path: str) -> torch.Tensor:
#     wav, sr = torchaudio.load(path)
#     if sr != WHISPER_SR:
#         wav = T.Resample(sr, WHISPER_SR)(wav)
#     if wav.shape[0] > 1:
#         wav = wav.mean(0, keepdim=True)
#     return wav.squeeze(0)   # (T,)


# @torch.no_grad()
# def embed_segment(seg_wav: torch.Tensor, model, fe, device) -> torch.Tensor:
#     """Returns mean-pooled embedding (512,) for one audio segment."""
#     T_seg = seg_wav.shape[0]
#     if T_seg == 0:
#         return torch.zeros(512)

#     all_embs = []
#     start = 0
#     while start < T_seg:
#         chunk = seg_wav[start : start + WHISPER_WIN_SAMP]
#         if chunk.shape[0] < WHISPER_WIN_SAMP:
#             chunk = torch.nn.functional.pad(
#                 chunk, (0, WHISPER_WIN_SAMP - chunk.shape[0])
#             )
#         feats = fe(
#             chunk.numpy(),
#             sampling_rate=WHISPER_SR,
#             return_tensors="pt",
#         ).input_features.to(device, dtype=torch.float32)

#         emb = model.encode(feats).squeeze(0).cpu()   # (1500, 512)

#         # Only keep frames for actual audio
#         actual = min(WHISPER_WIN_SAMP, T_seg - start)
#         n_frames = int(1500 * actual / WHISPER_WIN_SAMP)
#         all_embs.append(emb[:n_frames])

#         start += WHISPER_WIN_SAMP
#         if start >= T_seg:
#             break

#     emb_full = torch.cat(all_embs, dim=0)   # (N, 512)
#     return emb_full.mean(dim=0)              # (512,)


# # ── Main ──────────────────────────────────────────────────────────
# def extract(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info("Device: %s", device)

#     ttm_dir  = Path(args.ttm_dir)
#     wave_dir = Path(args.wave_dir)
#     out_dir  = Path(args.output) / args.split
#     out_dir.mkdir(parents=True, exist_ok=True)

#     fe    = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
#     model = build_model(args.checkpoint, device)

#     # Get clip list
#     if args.split in ("train", "val"):
#         split_file = Path(args.starter_code) / "data/split" / f"{args.split}.list"
#         with open(split_file) as f:
#             clip_uids = [l.strip() for l in f if l.strip()]
#     else:
#         clip_uids = [f.stem for f in ttm_dir.glob("*.json")]

#     logger.info("Processing %d clips for split=%s", len(clip_uids), args.split)

#     skipped = saved = seg_pos = seg_neg = 0

#     for uid in tqdm(clip_uids, desc=f"[{args.split}]"):

#         ttm_path = ttm_dir / f"{uid}.json"
#         wav_path = wave_dir / f"{uid}.wav"

#         if not ttm_path.exists() or not wav_path.exists():
#             skipped += 1
#             continue

#         try:
#             waveform = load_wav(str(wav_path))
#         except Exception as e:
#             logger.error("Audio load failed %s: %s", uid, e)
#             skipped += 1
#             continue

#         total_samples = waveform.shape[0]

#         with open(ttm_path) as f:
#             ttm_data = json.load(f)

#         for seg in ttm_data:
#             person_id   = int(seg["label"])
#             start_frame = int(seg["start_frame"])
#             end_frame   = int(seg["end_frame"])
#             seg_length  = end_frame - start_frame + 1

#             # ── TTM label ────────────────────────────────────────
#             if "tags" in seg and int(seg["tags"]) == 1:
#                 ttm_label = 1
#             else:
#                 ttm_label = 0

#             # ── Skip conditions ──────────────────────────────────
#             if person_id == 0:       # camera wearer
#                 continue
#             if seg_length < 3:       # too short
#                 continue

#             # ── Check if already extracted ───────────────────────
#             out_name = f"{uid}_{person_id}_{start_frame}_{end_frame}.json"
#             out_path = out_dir / out_name
#             if out_path.exists():
#                 saved += 1
#                 continue

#             # ── Cut audio ────────────────────────────────────────
#             onset  = max(0, int(start_frame / VIDEO_FPS * WHISPER_SR))
#             offset = min(total_samples, int(end_frame / VIDEO_FPS * WHISPER_SR))
#             if offset <= onset:
#                 continue
#             seg_wav = waveform[onset:offset]

#             # ── Embed ────────────────────────────────────────────
#             try:
#                 emb = embed_segment(seg_wav, model, fe, device)   # (512,)
#             except Exception as e:
#                 logger.error("Embed failed %s %d-%d: %s",
#                              uid, start_frame, end_frame, e)
#                 continue

#             # ── Save one JSON per segment ─────────────────────────
#             seg_data = {
#                 "uid":         uid,
#                 "person_id":   person_id,
#                 "label":       ttm_label,
#                 "tag":         int(seg.get("tags", 0)),
#                 "start_frame": start_frame,
#                 "end_frame":   end_frame,
#                 "duration_sec": round(seg_length / VIDEO_FPS, 3),
#                 "embedding":   emb.tolist(),    # 512 floats
#             }
#             with open(out_path, "w") as f:
#                 json.dump(seg_data, f)

#             saved += 1
#             if ttm_label == 1:
#                 seg_pos += 1
#             else:
#                 seg_neg += 1

#     logger.info(
#         "Done. Files saved=%d  Clips skipped=%d  "
#         "Pos=%d  Neg=%d  Pos_ratio=%.1f%%",
#         saved, skipped, seg_pos, seg_neg,
#         100 * seg_pos / max(seg_pos + seg_neg, 1),
#     )


# # ── CLI ───────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--ttm_dir",
#         default="/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/result_TTM")
#     p.add_argument("--wave_dir",
#         default="/DATA/G17/Data/wave")
#     p.add_argument("--output",
#         default="/DATA/G17/outputs/ttm_segment_embeddings")
#     p.add_argument("--checkpoint",
#         default="/DATA/G17/outputs/ttm_audio/checkpoints/ckpt_best.pt")
#     p.add_argument("--starter_code",
#         default="/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code")
#     p.add_argument("--split",
#         default="train", choices=["train", "val", "all"])
#     p.add_argument("--device", default="cuda")
#     extract(p.parse_args())

# import os
# import json
# import logging
# import argparse
# import warnings
# from pathlib import Path
# import torch
# import torchaudio
# import torchaudio.transforms as T
# from tqdm import tqdm
# from transformers import WhisperFeatureExtractor
# from audio_model import WhisperTTM

# warnings.filterwarnings("ignore")
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s – %(message)s")
# logger = logging.getLogger("seg_extract")

# VIDEO_FPS = 30
# WHISPER_SR = 16_000
# WHISPER_WIN_SAMP = int(30.0 * WHISPER_SR)

# def build_model(checkpoint_path, device):
#     model = WhisperTTM(
#         model_name="openai/whisper-large-v3",
#         freeze_encoder_layers=28,
#         projection_dim=512,
#         num_classes=2,
#     )
#     if checkpoint_path and Path(checkpoint_path).exists():
#         logger.info(f"Loading checkpoint: {checkpoint_path}")
#         ckpt = torch.load(checkpoint_path, map_location="cpu")
#         model.load_state_dict(ckpt.get("model", ckpt), strict=False)
#     model.float().eval().to(device)
#     return model

# def load_wav(path):
#     wav, sr = torchaudio.load(path)
#     if sr != WHISPER_SR:
#         wav = T.Resample(sr, WHISPER_SR)(wav)
#     return wav.mean(0) if wav.shape[0] > 1 else wav.squeeze(0)

# @torch.no_grad()
# def embed_segment(seg_wav, model, fe, device):
#     T_seg = seg_wav.shape[0]
#     if T_seg == 0: return torch.zeros(512)
    
#     # Whisper expects 30s chunks
#     if T_seg < WHISPER_WIN_SAMP:
#         chunk = torch.nn.functional.pad(seg_wav, (0, WHISPER_WIN_SAMP - T_seg))
#     else:
#         chunk = seg_wav[:WHISPER_WIN_SAMP]

#     feats = fe(chunk.numpy(), sampling_rate=WHISPER_SR, return_tensors="pt").input_features.to(device)
#     emb = model.encode(feats) 
    
#     # Pool to 512-dim vector
#     if len(emb.shape) == 3:
#         emb = emb.mean(dim=1) 
#     return emb.squeeze(0).cpu()

# def extract(args):
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     out_dir = Path(args.output)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
#     model = build_model(args.checkpoint, device)

#     ttm_dir = Path(args.ttm_dir)
#     wave_dir = Path(args.wave_dir)
#     master_data_list = []

#     json_files = list(ttm_dir.glob("*.json"))
    
#     for ttm_path in tqdm(json_files, desc="Processing Clips"):
#         uid = ttm_path.stem
#         wav_path = wave_dir / f"{uid}.wav"
#         if not wav_path.exists(): continue

#         waveform = load_wav(str(wav_path))
#         with open(ttm_path) as f:
#             ttm_data = json.load(f)

#         for seg in ttm_data:
#             person_id = seg.get("label", 0)
#             start_f = seg["start_frame"]
#             end_f = seg["end_frame"]
            
#             # Logic: tags present AND tags==1 -> TTM label 1
#             tag_val = seg.get("tags", seg.get("tag", 0))
#             label_ttm = 1 if tag_val == 1 else 0

#             onset = int(start_f / VIDEO_FPS * WHISPER_SR)
#             offset = int(end_f / VIDEO_FPS * WHISPER_SR)
#             seg_wav = waveform[onset:offset]

#             try:
#                 emb = embed_segment(seg_wav, model, fe, device)
#                 master_data_list.append({
#                     "uid": uid,
#                     "person_id": person_id,
#                     "label": label_ttm,
#                     "tag": tag_val,
#                     "start_frame": start_f,
#                     "end_frame": end_f,
#                     "embedding": emb.tolist()
#                 })
#             except Exception as e:
#                 logger.error(f"Error in {uid}: {e}")

#     # Final Single Output
#     with open(out_dir / "all_audio_embeddings.json", "w") as f:
#         json.dump(master_data_list, f)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ttm_dir", default="/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/result_TTM")
#     parser.add_argument("--wave_dir", default="/DATA/G17/Data/wave")
#     parser.add_argument("--output", default="/DATA/G17/outputs/ttm_segment_embeddings")
#     parser.add_argument("--checkpoint", default="/DATA/G17/outputs/ttm_audio/checkpoints/ckpt_best.pt")
#     parser.add_argument("--device", default="cuda") # Added to fix your error
#     parser.add_argument("--split", default="train")  # Added for compatibility
#     extract(parser.parse_args())


import os
import json
import logging
import argparse
import warnings
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from transformers import WhisperFeatureExtractor
from audio_model import WhisperTTM

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seg_extract")

VIDEO_FPS        = 30
WHISPER_SR       = 16_000
WHISPER_WIN_SAMP = int(30.0 * WHISPER_SR)   # 480000


def build_model(checkpoint_path, device):
    model = WhisperTTM(
        model_name="openai/whisper-large-v3",
        freeze_encoder_layers=28,
        projection_dim=512,
        num_classes=2,
    )
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    else:
        logger.info("No checkpoint – using raw Whisper weights")
    model.float().eval().to(device)
    logger.info("Model loaded and ready on %s", device)
    return model


def load_wav(path):
    wav, sr = torchaudio.load(path)
    if sr != WHISPER_SR:
        wav = T.Resample(sr, WHISPER_SR)(wav)
    return wav.mean(0) if wav.shape[0] > 1 else wav.squeeze(0)


@torch.no_grad()
def embed_segment(seg_wav, model, fe, device):
    T_seg = seg_wav.shape[0]
    if T_seg == 0:
        return torch.zeros(512)

    if T_seg < WHISPER_WIN_SAMP:
        chunk = torch.nn.functional.pad(seg_wav, (0, WHISPER_WIN_SAMP - T_seg))
    else:
        chunk = seg_wav[:WHISPER_WIN_SAMP]

    feats = fe(
        chunk.numpy(),
        sampling_rate=WHISPER_SR,
        return_tensors="pt",
    ).input_features.to(device, dtype=torch.float32)   # ← fp32 fix

    emb = model.encode(feats)          # (1, 1500, 512)
    if len(emb.shape) == 3:
        emb = emb.mean(dim=1)          # (1, 512)
    return emb.squeeze(0).cpu()        # (512,)


def extract(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_audio_embeddings.json"

    fe    = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    model = build_model(args.checkpoint, device)

    ttm_dir  = Path(args.ttm_dir)
    wave_dir = Path(args.wave_dir)

    json_files = sorted(ttm_dir.glob("*.json"))
    logger.info("Total clip JSON files found: %d", len(json_files))

    master_data_list = []
    clips_done   = 0
    clips_skip   = 0
    seg_pos      = 0
    seg_neg      = 0
    seg_skip     = 0

    for ttm_path in tqdm(json_files, desc="Processing Clips"):
        uid      = ttm_path.stem
        wav_path = wave_dir / f"{uid}.wav"

        if not wav_path.exists():
            logger.warning("WAV not found – skipping: %s", uid)
            clips_skip += 1
            continue

        try:
            waveform = load_wav(str(wav_path))
        except Exception as e:
            logger.error("Failed to load WAV %s: %s", uid, e)
            clips_skip += 1
            continue

        total_samples = waveform.shape[0]

        with open(ttm_path) as f:
            ttm_data = json.load(f)

        clip_segs = 0
        for seg in ttm_data:
            person_id   = int(seg.get("label", 0))
            start_frame = int(seg["start_frame"])
            end_frame   = int(seg["end_frame"])
            seg_length  = end_frame - start_frame + 1

            # TTM label
            tag_val   = seg.get("tags", seg.get("tag", 0))
            label_ttm = 1 if tag_val == 1 else 0

            # Skip camera wearer and too-short
            if person_id == 0 or seg_length < 3:
                seg_skip += 1
                continue

            onset  = max(0, int(start_frame / VIDEO_FPS * WHISPER_SR))
            offset = min(total_samples, int(end_frame / VIDEO_FPS * WHISPER_SR))
            if offset <= onset:
                seg_skip += 1
                continue

            seg_wav = waveform[onset:offset]

            try:
                emb = embed_segment(seg_wav, model, fe, device)
                master_data_list.append({
                    "uid":         uid,
                    "label":       person_id,    # person identity
                    "tags":        label_ttm,    # 0 or 1 (TTM label)
                    "start_frame": start_frame,
                    "end_frame":   end_frame,
                    "embedding":   emb.tolist(), # 512 floats
                })
                clip_segs += 1
                if label_ttm == 1:
                    seg_pos += 1
                else:
                    seg_neg += 1
            except Exception as e:
                logger.error("Embed failed %s seg %d-%d: %s",
                             uid, start_frame, end_frame, e)
                seg_skip += 1

        clips_done += 1

        # ── Progress every 50 clips ──────────────────────────────
        if clips_done % 50 == 0:
            total_segs = seg_pos + seg_neg
            logger.info(
                "Progress: %d/%d clips  |  segs so far: %d  "
                "(pos=%d %.1f%%  neg=%d %.1f%%)",
                clips_done, len(json_files),
                total_segs,
                seg_pos, 100 * seg_pos / max(total_segs, 1),
                seg_neg, 100 * seg_neg / max(total_segs, 1),
            )

    # ── Save single combined JSON ────────────────────────────────
    logger.info("Saving all_audio_embeddings.json ...")
    with open(out_path, "w") as f:
        json.dump(master_data_list, f)

    total_segs = seg_pos + seg_neg
    logger.info("=" * 55)
    logger.info("EXTRACTION COMPLETE")
    logger.info("  Output file     : %s", out_path)
    logger.info("  File size       : %.1f MB", out_path.stat().st_size / 1e6)
    logger.info("  Clips processed : %d", clips_done)
    logger.info("  Clips skipped   : %d", clips_skip)
    logger.info("  Total segments  : %d", total_segs)
    logger.info("  TTM=1 (pos)     : %d  (%.1f%%)", seg_pos, 100 * seg_pos / max(total_segs, 1))
    logger.info("  TTM=0 (neg)     : %d  (%.1f%%)", seg_neg, 100 * seg_neg / max(total_segs, 1))
    logger.info("  Segs skipped    : %d", seg_skip)
    logger.info("  Embedding shape : (512,) per segment")
    logger.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttm_dir",
        default="/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/result_TTM")
    parser.add_argument("--wave_dir",
        default="/DATA/G17/Data/wave")
    parser.add_argument("--output",
        default="/DATA/G17/outputs/ttm_segment_embeddings")
    parser.add_argument("--checkpoint",
        default="/DATA/G17/outputs/ttm_audio/checkpoints/ckpt_best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="all")
    extract(parser.parse_args())