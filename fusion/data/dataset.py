import json
import random

import torch
from torch.utils.data import Dataset


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class FusionDataset(Dataset):
    def __init__(self, json_path, split="train", val_ratio=0.2, seed=42):
        self.json_path = json_path
        self.split = split

        with open(json_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
            raise ValueError(f"Expected a JSON object with 'segments' list in {json_path}")

        all_samples = []
        for seg in data["segments"]:
            v = seg.get("visual_embedding", [])
            a = seg.get("audio_embedding", [])

            if not isinstance(v, list) or not isinstance(a, list):
                continue
            if len(v) == 0 or len(a) == 0:
                continue

            label = 1 if _to_int(seg.get("label"), default=0) == 1 else 0
            all_samples.append((v, a, label))

        if not all_samples:
            raise RuntimeError(f"No usable samples found in {json_path}")

        rng = random.Random(seed)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)

        split_idx = int(len(indices) * (1.0 - val_ratio))
        split_idx = min(max(split_idx, 1), len(indices) - 1)

        if split == "train":
            chosen = indices[:split_idx]
        elif split == "val":
            chosen = indices[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.samples = [all_samples[i] for i in chosen]

        if not self.samples:
            raise RuntimeError(f"Split '{split}' has no samples from {json_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        v_emb, a_emb, label = self.samples[idx]

        visual = torch.tensor(v_emb, dtype=torch.float32).unsqueeze(0)
        audio = torch.tensor(a_emb, dtype=torch.float32).unsqueeze(0)
        labels = torch.tensor([label], dtype=torch.float32)

        return visual, audio, labels