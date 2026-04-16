import numpy as np
import torch
import torchaudio
import os

class TimeGridGenerator:
    def __init__(self, clip_duration, stride=0.2, video_fps=25):
        self.stride = stride
        self.video_fps = video_fps
        self.clip_duration = clip_duration
        self.num_bins = int(np.ceil(clip_duration / stride))
        self.time_centers = np.arange(self.num_bins) * stride + stride / 2
        self.time_centers = self.time_centers[self.time_centers < clip_duration]
        self.num_bins = len(self.time_centers)

    def get_audio_window(self, waveform, sample_rate, time_center, window_duration):
        window_samples = int(window_duration * sample_rate)
        center_sample = int(time_center * sample_rate)
        start = max(0, center_sample - window_samples // 2)
        end = min(waveform.shape[1], start + window_samples)
        chunk = waveform[:, start:end]
        if chunk.shape[1] < window_samples:
            chunk = torch.nn.functional.pad(chunk, (0, window_samples - chunk.shape[1]))
        return chunk

    def get_frame_path(self, frames_dir, time_center):
        frame_idx = int(round(time_center * self.video_fps))
        # Ego4D frames may be stored with either img_ or frame_ prefixes.
        possible_names = [
            f"img_{frame_idx+1:05d}.jpg",
            f"frame_{frame_idx:06d}.jpg",
            f"{frame_idx:06d}.jpg"
        ]
        for name in possible_names:
            path = os.path.join(frames_dir, name)
            if os.path.exists(path):
                return path
        # Fallback to the most common naming convention when no file has been found.
        return os.path.join(frames_dir, possible_names[0])
