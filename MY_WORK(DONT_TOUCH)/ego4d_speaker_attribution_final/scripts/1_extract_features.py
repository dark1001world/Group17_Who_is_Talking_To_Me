#!/usr/bin/env python3
"""
Extract aligned audio-visual features for Ego4D Talking-to-Me clips.

For each clip this script saves:
  - <clip_uid>_audio.npy   shape [T, audio_dim]
  - <clip_uid>_visual.npy  shape [T, N, visual_dim]
  - <clip_uid>_mask.npy    shape [T, N]
  - <clip_uid>_labels.npy  shape [T, N]

Optional debug files:
  - <clip_uid>_times.npy
  - <clip_uid>_person_ids.npy
"""

import json
import os
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from json import JSONDecodeError

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.embedding_extractor import AudioEmbeddingExtractor
from src.utils.alignment import TimeGridGenerator
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.visual.feature_extractor import VisualFeatureExtractor


class ExtractionAbort(RuntimeError):
    pass


def resolve_device(requested_device):
    if requested_device == "cuda":
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested_device


def stream_json_array(path, chunk_size=1 << 20):
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as handle:
        buffer = ""
        started = False

        while True:
            if not buffer:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                buffer = chunk

            while True:
                buffer = buffer.lstrip()

                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    buffer = buffer[1:]
                    started = True
                    continue

                if not buffer:
                    break

                if buffer[0] == "]":
                    return
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue

                try:
                    item, index = decoder.raw_decode(buffer)
                except JSONDecodeError:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        raise
                    buffer += chunk
                    continue

                yield item
                buffer = buffer[index:]


def normalize_bbox(bbox):
    if bbox is None or len(bbox) != 4:
        return None

    try:
        x, y, w, h = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    if w <= 1 or h <= 1:
        return None

    return [x, y, x + w, y + h]


def build_annotation_index(annotation_file, clip_uids, logger):
    clip_set = set(clip_uids)
    clip_annotations = {
        clip_uid: {"frames": defaultdict(list), "person_ids": set(), "stats": Counter()}
        for clip_uid in clip_set
    }

    logger.info("Indexing annotations from %s", annotation_file)
    for item in tqdm(stream_json_array(annotation_file), desc="Reading annotations"):
        clip_uid = item.get("clip_uid")
        if clip_uid not in clip_set:
            continue

        frame_idx = item.get("frame")
        person_id = str(item.get("person_id"))
        ttm_label = int(item.get("ttm_label", 0))
        bbox = normalize_bbox(item.get("bbox"))

        clip_annotations[clip_uid]["stats"]["rows_seen"] += 1
        if frame_idx is None:
            clip_annotations[clip_uid]["stats"]["missing_frame_index"] += 1
            continue
        if bbox is None:
            clip_annotations[clip_uid]["stats"]["invalid_bbox_in_json"] += 1
            continue

        frame_idx = int(frame_idx)
        clip_annotations[clip_uid]["frames"][frame_idx].append(
            {"person_id": person_id, "bbox": bbox, "label": ttm_label}
        )
        clip_annotations[clip_uid]["person_ids"].add(person_id)

    for clip_uid, payload in clip_annotations.items():
        payload["frame_indices"] = sorted(payload["frames"].keys())
        payload["person_ids"] = sorted(
            payload["person_ids"], key=lambda value: int(value) if value.isdigit() else value
        )

    return clip_annotations


def find_nearest_annotated_frame(frame_indices, target_frame, max_delta):
    if not frame_indices:
        return None

    pos = bisect_left(frame_indices, target_frame)
    candidates = []
    if pos < len(frame_indices):
        candidates.append(frame_indices[pos])
    if pos > 0:
        candidates.append(frame_indices[pos - 1])

    if not candidates:
        return None

    nearest = min(candidates, key=lambda value: abs(value - target_frame))
    if abs(nearest - target_frame) > max_delta:
        return None
    return nearest


def resolve_frame_path(frames_dir, frame_idx):
    return os.path.join(frames_dir, f"img_{frame_idx + 1:05d}.jpg")


def crop_face(frame_rgb, bbox, face_size):
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width - 1, int(np.floor(x1))))
    y1 = max(0, min(height - 1, int(np.floor(y1))))
    x2 = max(0, min(width, int(np.ceil(x2))))
    y2 = max(0, min(height, int(np.ceil(y2))))

    if x2 <= x1 or y2 <= y1:
        return None, "bbox_outside_image"
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None, "bbox_too_small_after_clamp"

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None, "empty_crop"

    resized = cv2.resize(crop, (face_size, face_size), interpolation=cv2.INTER_LINEAR)
    return resized, None


def summarize_counter(counter):
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def main():
    config = load_config("configs/default.yaml")
    logger = setup_logger("extract_features", config["logging"]["log_dir"])

    extraction_cfg = config.get("extraction", {})
    stride = float(config["alignment"]["stride"])
    window_size = float(config["alignment"]["window_size"])
    video_fps = int(config["alignment"]["video_fps"])
    frame_tolerance = int(extraction_cfg.get("frame_tolerance", 3))
    min_valid_tracks = int(extraction_cfg.get("min_valid_tracks", 1))
    min_valid_frames_per_track = int(extraction_cfg.get("min_valid_frames_per_track", 1))
    require_positive_labels = bool(extraction_cfg.get("require_positive_labels", True))
    save_debug_arrays = bool(extraction_cfg.get("save_debug_arrays", True))
    max_visual_errors_per_clip = int(extraction_cfg.get("max_visual_errors_per_clip", 5))
    max_visual_errors_total = int(extraction_cfg.get("max_visual_errors_total", 25))

    device = resolve_device(config["training"]["device"])
    if device != config["training"]["device"]:
        logger.warning(
            "Requested device %s is unavailable; falling back to %s",
            config["training"]["device"],
            device,
        )

    audio_dir = config["data"]["audio_dir"]
    frames_root = config["data"]["frames_dir"]
    annotation_file = config["data"]["annotation_file"]
    output_dir = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    audio_files = sorted(name for name in os.listdir(audio_dir) if name.endswith(".wav"))
    if not audio_files:
        logger.error("No .wav files found in %s", audio_dir)
        return

    clip_uids = [os.path.splitext(name)[0] for name in audio_files]
    annotation_index = build_annotation_index(annotation_file, clip_uids, logger)

    logger.info("Initializing feature extractors on %s", device)
    audio_extractor = AudioEmbeddingExtractor(
        model_name=config["audio"]["model_name"].split("/")[-1].split("-")[0],
        device=device,
    )
    visual_extractor = VisualFeatureExtractor(config, device=device)

    total_saved = 0
    total_skipped = 0
    total_visual_errors = 0

    for audio_file in tqdm(audio_files, desc="Processing clips"):
        clip_uid = os.path.splitext(audio_file)[0]
        audio_path = os.path.join(audio_dir, audio_file)
        clip_frames_dir = os.path.join(frames_root, clip_uid)
        clip_info = annotation_index.get(clip_uid, {})
        clip_reason_counts = Counter()

        if not os.path.isdir(clip_frames_dir):
            logger.warning("%s: missing frames directory %s", clip_uid, clip_frames_dir)
            total_skipped += 1
            continue

        frame_map = clip_info.get("frames", {})
        frame_indices = clip_info.get("frame_indices", [])
        person_ids = clip_info.get("person_ids", [])

        if not frame_indices:
            stats = summarize_counter(clip_info.get("stats", Counter()))
            logger.warning("%s: no usable annotations found (%s)", clip_uid, stats)
            total_skipped += 1
            continue

        if not person_ids:
            logger.warning("%s: no person IDs found in annotations", clip_uid)
            total_skipped += 1
            continue

        try:
            audio_features, total_samples = audio_extractor.extract_features_full(audio_path)
        except Exception as exc:
            logger.exception("%s: audio feature extraction failed: %s", clip_uid, exc)
            total_skipped += 1
            continue

        clip_duration = total_samples / float(audio_extractor.sample_rate)
        grid = TimeGridGenerator(clip_duration, stride=stride, video_fps=video_fps)
        if grid.num_bins == 0:
            logger.warning("%s: empty time grid for clip duration %.3fs", clip_uid, clip_duration)
            total_skipped += 1
            continue

        person_to_index = {person_id: index for index, person_id in enumerate(person_ids)}
        num_bins = grid.num_bins
        num_tracks = len(person_ids)
        audio_dim = int(audio_features.shape[1])
        visual_dim = (
            int(config["visual"]["proj_vit_dim"])
            + int(config["visual"]["proj_reid_dim"])
            + int(config["visual"]["proj_lip_dim"])
        )

        audio_array = np.zeros((num_bins, audio_dim), dtype=np.float32)
        visual_array = np.zeros((num_bins, num_tracks, visual_dim), dtype=np.float32)
        mask_array = np.zeros((num_bins, num_tracks), dtype=bool)
        label_array = np.zeros((num_bins, num_tracks), dtype=np.float32)

        for bin_index, time_center in enumerate(grid.time_centers):
            audio_array[bin_index] = audio_extractor.get_embedding_at_time(
                audio_features,
                total_samples,
                time_center,
                window_size,
            ).astype(np.float32)

            target_frame = int(round(time_center * video_fps))
            annotated_frame = find_nearest_annotated_frame(frame_indices, target_frame, frame_tolerance)
            if annotated_frame is None:
                clip_reason_counts["no_annotation_near_bin"] += 1
                continue

            frame_path = resolve_frame_path(clip_frames_dir, annotated_frame)
            if not os.path.exists(frame_path):
                clip_reason_counts["missing_frame_file"] += 1
                continue

            frame_bgr = cv2.imread(frame_path)
            if frame_bgr is None:
                clip_reason_counts["unreadable_frame_file"] += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tracked_faces = []
            frame_annotations = frame_map.get(annotated_frame, [])

            for annotation in frame_annotations:
                face_crop, crop_reason = crop_face(
                    frame_rgb,
                    annotation["bbox"],
                    config["visual"]["face_size"],
                )
                if face_crop is None:
                    clip_reason_counts[crop_reason] += 1
                    continue

                tracked_faces.append({"id": annotation["person_id"], "face_crop": face_crop})

            if not tracked_faces:
                clip_reason_counts["no_valid_face_crops"] += 1
                continue

            try:
                visual_features = visual_extractor.extract_features(
                    frame_rgb,
                    tracked_faces,
                    time_center,
                )
            except Exception as exc:
                clip_reason_counts["visual_extraction_error"] += 1
                total_visual_errors += 1
                logger.error(
                    "%s: visual feature extraction failed at frame %s (%s): %s [clip_errors=%d total_errors=%d]",
                    clip_uid,
                    annotated_frame,
                    frame_path,
                    exc,
                    clip_reason_counts["visual_extraction_error"],
                    total_visual_errors,
                )

                if clip_reason_counts["visual_extraction_error"] >= max_visual_errors_per_clip:
                    raise ExtractionAbort(
                        f"{clip_uid}: aborting because visual extraction hit "
                        f"{clip_reason_counts['visual_extraction_error']} errors "
                        f"(limit={max_visual_errors_per_clip})"
                    ) from exc

                if total_visual_errors >= max_visual_errors_total:
                    raise ExtractionAbort(
                        "Aborting feature extraction because visual extraction errors reached "
                        f"{total_visual_errors} (limit={max_visual_errors_total})"
                    ) from exc
                continue

            if not visual_features:
                clip_reason_counts["empty_visual_feature_dict"] += 1
                continue

            for annotation in frame_annotations:
                person_id = annotation["person_id"]
                feature = visual_features.get(person_id)
                if feature is None:
                    clip_reason_counts["missing_person_feature"] += 1
                    continue

                track_index = person_to_index[person_id]
                visual_array[bin_index, track_index] = feature.astype(np.float32)
                mask_array[bin_index, track_index] = True
                label_array[bin_index, track_index] = float(annotation["label"])

        valid_track_mask = mask_array.sum(axis=0) >= min_valid_frames_per_track
        valid_track_count = int(valid_track_mask.sum())

        if valid_track_count < min_valid_tracks:
            logger.warning(
                "%s: skipping clip because only %d tracks met min_valid_frames_per_track=%d (%s)",
                clip_uid,
                valid_track_count,
                min_valid_frames_per_track,
                summarize_counter(clip_reason_counts),
            )
            total_skipped += 1
            continue

        audio_array = audio_array
        visual_array = visual_array[:, valid_track_mask, :]
        mask_array = mask_array[:, valid_track_mask]
        label_array = label_array[:, valid_track_mask]
        kept_person_ids = np.array(
            [person_id for person_id, keep in zip(person_ids, valid_track_mask) if keep]
        )

        if require_positive_labels and not np.any(label_array):
            logger.warning(
                "%s: skipping clip because it has no positive labels after filtering (%s)",
                clip_uid,
                summarize_counter(clip_reason_counts),
            )
            total_skipped += 1
            continue

        base_path = os.path.join(output_dir, clip_uid)
        np.save(f"{base_path}_audio.npy", audio_array)
        np.save(f"{base_path}_visual.npy", visual_array)
        np.save(f"{base_path}_mask.npy", mask_array)
        np.save(f"{base_path}_labels.npy", label_array)

        if save_debug_arrays:
            np.save(f"{base_path}_times.npy", grid.time_centers.astype(np.float32))
            np.save(f"{base_path}_person_ids.npy", kept_person_ids)

        logger.info(
            "%s: saved T=%d, N=%d to %s (%s)",
            clip_uid,
            audio_array.shape[0],
            visual_array.shape[1],
            output_dir,
            summarize_counter(clip_reason_counts),
        )
        total_saved += 1

    logger.info("Feature extraction finished: saved=%d skipped=%d", total_saved, total_skipped)


if __name__ == "__main__":
    try:
        main()
    except ExtractionAbort as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
