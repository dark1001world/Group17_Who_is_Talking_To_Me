import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TTMSegment:
    clip_uid: str
    person_id: str
    label: int
    start_frame: int
    end_frame: int
    segment_index: int


def _read_split_file(split_file: str) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _normalize_audio(samples: np.ndarray, desired_rms: float = 0.1, eps: float = 1e-4) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32)
    rms = max(eps, float(np.sqrt(np.mean(samples ** 2))))
    return (samples * (desired_rms / rms)).astype(np.float32)


def _manual_image_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


class TTMBenchmarkDataset(Dataset):
    def __init__(
        self,
        split_file: str,
        video_dir: str,
        audio_dir: str,
        tracklet_dir: str,
        ttm_dir: str,
        *,
        fps: int = 30,
        sample_rate: int = 16000,
        crop_size: int = 224,
        min_frames: int = 15,
        max_frames: int = 150,
        frame_step: int = 1,
        min_valid_face_frames: int = 3,
        filter_missing_tracks: bool = True,
    ) -> None:
        self.split_file = split_file
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.tracklet_dir = tracklet_dir
        self.ttm_dir = ttm_dir
        self.fps = fps
        self.sample_rate = sample_rate
        self.crop_size = crop_size
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.frame_step = frame_step
        self.min_valid_face_frames = min_valid_face_frames
        self.filter_missing_tracks = filter_missing_tracks

        self.clip_uids = _read_split_file(split_file)
        self._bbox_cache: Dict[str, Dict[str, Dict[int, Tuple[float, float, float, float]]]] = {}
        self._audio_cache: Dict[str, np.ndarray] = {}
        self.segments = self._build_segments()

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        segment = self.segments[idx]
        bbox_map = self._load_bbox_map(segment.clip_uid)
        frame_ids = self._select_frame_ids(segment.start_frame, segment.end_frame)

        video_frames = []
        face_mask = []
        valid_face_frames = 0
        person_boxes = bbox_map.get(segment.person_id, {})

        for frame_id in frame_ids:
            bbox = person_boxes.get(frame_id)
            if bbox is None:
                video_frames.append(torch.zeros(3, self.crop_size, self.crop_size, dtype=torch.float32))
                face_mask.append(False)
                continue

            image_path = os.path.join(self.video_dir, segment.clip_uid, f"img_{frame_id:05d}.jpg")
            if not os.path.exists(image_path):
                video_frames.append(torch.zeros(3, self.crop_size, self.crop_size, dtype=torch.float32))
                face_mask.append(False)
                continue

            frame_bgr = cv2.imread(image_path)
            if frame_bgr is None:
                video_frames.append(torch.zeros(3, self.crop_size, self.crop_size, dtype=torch.float32))
                face_mask.append(False)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            face_crop = self._crop_face(frame_rgb, bbox)
            if face_crop is None:
                video_frames.append(torch.zeros(3, self.crop_size, self.crop_size, dtype=torch.float32))
                face_mask.append(False)
                continue

            video_frames.append(_manual_image_to_tensor(face_crop))
            face_mask.append(True)
            valid_face_frames += 1

        waveform = self._load_audio(segment.clip_uid)
        onset = int(segment.start_frame / self.fps * self.sample_rate)
        offset = int((segment.end_frame + 1) / self.fps * self.sample_rate)
        offset = max(offset, onset + 1)
        audio_crop = waveform[onset:offset]
        audio_crop = _normalize_audio(audio_crop)

        return {
            "video": torch.stack(video_frames, dim=0),
            "face_mask": torch.tensor(face_mask, dtype=torch.bool),
            "frame_ids": torch.tensor(frame_ids, dtype=torch.long),
            "audio": torch.from_numpy(audio_crop),
            "label": torch.tensor(segment.label, dtype=torch.float32),
            "metadata": {
                "clip_uid": segment.clip_uid,
                "person_id": segment.person_id,
                "segment_index": segment.segment_index,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "valid_face_frames": valid_face_frames,
            },
        }

    def _build_segments(self) -> List[TTMSegment]:
        segments: List[TTMSegment] = []
        for clip_uid in self.clip_uids:
            label_path = os.path.join(self.ttm_dir, f"{clip_uid}.json")
            if not os.path.exists(label_path):
                continue

            with open(label_path, "r", encoding="utf-8") as handle:
                clip_segments = json.load(handle)

            for segment_index, item in enumerate(clip_segments):
                person_id = str(item.get("label", ""))
                if not person_id or person_id == "0":
                    continue

                start_frame = int(item["start_frame"])
                end_frame = int(item["end_frame"])
                seg_length = end_frame - start_frame + 1
                if seg_length < self.min_frames:
                    continue

                label = 1 if "tags" in item else 0
                for sub_start, sub_end in self._chunk_segment(start_frame, end_frame):
                    if self.filter_missing_tracks and not self._has_face_frames(clip_uid, person_id, sub_start, sub_end):
                        continue

                    segments.append(
                        TTMSegment(
                            clip_uid=clip_uid,
                            person_id=person_id,
                            label=label,
                            start_frame=sub_start,
                            end_frame=sub_end,
                            segment_index=segment_index,
                        )
                    )
        return segments

    def _chunk_segment(self, start_frame: int, end_frame: int) -> List[Tuple[int, int]]:
        seg_length = end_frame - start_frame + 1
        if seg_length <= self.max_frames:
            return [(start_frame, end_frame)]

        chunks = []
        cursor = start_frame
        while cursor <= end_frame:
            sub_end = min(end_frame, cursor + self.max_frames - 1)
            if sub_end - cursor + 1 >= self.min_frames:
                chunks.append((cursor, sub_end))
            cursor += self.max_frames
        return chunks

    def _select_frame_ids(self, start_frame: int, end_frame: int) -> List[int]:
        frame_ids = list(range(start_frame, end_frame + 1, self.frame_step))
        if len(frame_ids) <= self.max_frames:
            return frame_ids

        sampled = np.linspace(0, len(frame_ids) - 1, num=self.max_frames)
        sampled = np.round(sampled).astype(int)
        return [frame_ids[index] for index in sampled]

    def _load_bbox_map(self, clip_uid: str) -> Dict[str, Dict[int, Tuple[float, float, float, float]]]:
        if clip_uid in self._bbox_cache:
            return self._bbox_cache[clip_uid]

        clip_dir = os.path.join(self.tracklet_dir, clip_uid)
        bbox_map: Dict[str, Dict[int, Tuple[float, float, float, float]]] = {}
        for track_path in glob.glob(os.path.join(clip_dir, "*.json")):
            with open(track_path, "r", encoding="utf-8") as handle:
                track_frames = json.load(handle)

            for frame in track_frames:
                person_id = str(frame.get("Person ID", ""))
                if not person_id:
                    continue
                frame_id = int(round(float(frame["frameNumber"])))
                x = float(frame["x"])
                y = float(frame["y"])
                w = float(frame["width"])
                h = float(frame["height"])
                if w <= 0 or h <= 0:
                    continue
                bbox_map.setdefault(person_id, {})[frame_id] = (x, y, x + w, y + h)

        self._bbox_cache[clip_uid] = bbox_map
        return bbox_map

    def _has_face_frames(self, clip_uid: str, person_id: str, start_frame: int, end_frame: int) -> bool:
        bbox_map = self._load_bbox_map(clip_uid)
        person_boxes = bbox_map.get(person_id, {})
        valid = 0
        for frame_id in range(start_frame, end_frame + 1):
            if frame_id in person_boxes:
                valid += 1
            if valid >= self.min_valid_face_frames:
                return True
        return False

    def _load_audio(self, clip_uid: str) -> np.ndarray:
        if clip_uid in self._audio_cache:
            return self._audio_cache[clip_uid]

        audio_path = os.path.join(self.audio_dir, f"{clip_uid}.wav")
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        audio = waveform.squeeze(0).cpu().numpy().astype(np.float32)
        self._audio_cache[clip_uid] = audio
        return audio

    def _crop_face(self, frame_rgb: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        height, width = frame_rgb.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width - 1, int(np.floor(x1))))
        y1 = max(0, min(height - 1, int(np.floor(y1))))
        x2 = max(0, min(width, int(np.ceil(x2))))
        y2 = max(0, min(height, int(np.ceil(y2))))
        if x2 <= x1 or y2 <= y1:
            return None
        face = frame_rgb[y1:y2, x1:x2]
        if face.size == 0:
            return None
        return cv2.resize(face, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)


def ttm_benchmark_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    max_frames = max(item["video"].shape[0] for item in batch)
    max_audio = max(item["audio"].shape[0] for item in batch)

    videos = []
    face_masks = []
    frame_masks = []
    frame_ids = []
    audios = []
    audio_masks = []
    labels = []
    metadata = []

    for item in batch:
        video = item["video"]
        face_mask = item["face_mask"]
        frames = item["frame_ids"]
        audio = item["audio"]

        pad_frames = max_frames - video.shape[0]
        pad_audio = max_audio - audio.shape[0]

        if pad_frames > 0:
            video = torch.cat(
                [video, torch.zeros(pad_frames, *video.shape[1:], dtype=video.dtype)],
                dim=0,
            )
            face_mask = torch.cat([face_mask, torch.zeros(pad_frames, dtype=torch.bool)], dim=0)
            frames = torch.cat([frames, torch.zeros(pad_frames, dtype=torch.long)], dim=0)

        if pad_audio > 0:
            audio = torch.cat([audio, torch.zeros(pad_audio, dtype=audio.dtype)], dim=0)

        videos.append(video)
        face_masks.append(face_mask)
        frame_masks.append(
            torch.tensor([True] * item["video"].shape[0] + [False] * pad_frames, dtype=torch.bool)
        )
        frame_ids.append(frames)
        audios.append(audio)
        audio_masks.append(
            torch.tensor([True] * item["audio"].shape[0] + [False] * pad_audio, dtype=torch.bool)
        )
        labels.append(item["label"])
        metadata.append(item["metadata"])

    return {
        "video": torch.stack(videos, dim=0),
        "face_mask": torch.stack(face_masks, dim=0),
        "frame_mask": torch.stack(frame_masks, dim=0),
        "frame_ids": torch.stack(frame_ids, dim=0),
        "audio": torch.stack(audios, dim=0),
        "audio_mask": torch.stack(audio_masks, dim=0),
        "label": torch.stack(labels, dim=0),
        "metadata": metadata,
    }
