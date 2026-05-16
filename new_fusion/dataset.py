import torch
import os
import json
from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self, audio_dir, visual_dir, annotation_json):
        self.audio_dir = audio_dir
        self.visual_dir = visual_dir
        
        # Load labels into a dictionary for O(1) lookup
        with open(annotation_json, "r") as f:
            data = json.load(f)
            
        self.labels = {}
        for r in data:
            uid = r['clip_uid']
            pid = r['person_id']
            fid = r['frame']
            # Match the "Universal Key" format
            key = f"{uid}_{pid}_{int(fid):06d}.pt"
            self.labels[key] = r['ttm_label']
            
        # Only keep files that exist in both directories AND have a label
        audio_files = set(os.listdir(audio_dir))
        visual_files = set(os.listdir(visual_dir))
        
        self.filenames = sorted(list(audio_files.intersection(visual_files).intersection(self.labels.keys())))
        print(f"Found {len(self.filenames)} perfectly aligned audio-visual pairs.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        audio_feat = torch.load(os.path.join(self.audio_dir, fname), weights_only=True)
        visual_feat = torch.load(os.path.join(self.visual_dir, fname), weights_only=True)
        label = self.labels[fname]
        
        # Squeeze out any empty dimensions if they were saved with them
        audio_feat = audio_feat.view(-1)
        visual_feat = visual_feat.view(-1)
        
        return audio_feat, visual_feat, torch.tensor(label, dtype=torch.long)