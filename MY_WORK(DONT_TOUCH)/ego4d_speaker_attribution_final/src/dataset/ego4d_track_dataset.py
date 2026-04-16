import torch
from torch.utils.data import Dataset
import numpy as np
import os

class Ego4DTrackDataset(Dataset):
    def __init__(self, data_dir, max_tracks=10, max_seq_len=300):
        self.data_dir = data_dir
        self.max_tracks = max_tracks
        self.max_seq_len = max_seq_len
        self.samples = [f.replace('_audio.npy', '') for f in os.listdir(data_dir)
                        if f.endswith('_audio.npy')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base = self.samples[idx]
        audio = np.load(os.path.join(self.data_dir, f"{base}_audio.npy"))
        visual = np.load(os.path.join(self.data_dir, f"{base}_visual.npy"))
        mask = np.load(os.path.join(self.data_dir, f"{base}_mask.npy"))
        labels = np.load(os.path.join(self.data_dir, f"{base}_labels.npy"))

        T, N = visual.shape[:2]
        if N > self.max_tracks:
            presence = mask.sum(axis=0)
            top_indices = np.argsort(presence)[-self.max_tracks:]
            visual = visual[:, top_indices, :]
            mask = mask[:, top_indices]
            labels = labels[:, top_indices]
            N = self.max_tracks

        if N < self.max_tracks:
            pad_tracks = self.max_tracks - N
            visual = np.pad(visual, ((0,0),(0,pad_tracks),(0,0)), mode='constant')
            mask = np.pad(mask, ((0,0),(0,pad_tracks)), mode='constant')
            labels = np.pad(labels, ((0,0),(0,pad_tracks)), mode='constant')

        if T > self.max_seq_len:
            audio = audio[:self.max_seq_len]
            visual = visual[:self.max_seq_len]
            mask = mask[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        elif T < self.max_seq_len:
            pad_len = self.max_seq_len - T
            audio = np.pad(audio, ((0,pad_len),(0,0)), mode='constant')
            visual = np.pad(visual, ((0,pad_len),(0,0),(0,0)), mode='constant')
            mask = np.pad(mask, ((0,pad_len),(0,0)), mode='constant')
            labels = np.pad(labels, ((0,pad_len),(0,0)), mode='constant')

        return {
            'audio': torch.FloatTensor(audio),
            'visual': torch.FloatTensor(visual),
            'track_mask': torch.BoolTensor(mask),
            'labels': torch.FloatTensor(labels)
        }
