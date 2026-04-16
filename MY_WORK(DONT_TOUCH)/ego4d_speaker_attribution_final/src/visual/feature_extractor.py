import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from .lip_encoder import LipMotionEncoder
from .reid_model import FaceReIDModel

class VisualFeatureExtractor(nn.Module):
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.device = device
        self.config = config

        self.vit_processor = ViTImageProcessor.from_pretrained(config['visual']['vit_model'])
        self.vit_model = ViTModel.from_pretrained(config['visual']['vit_model']).to(device)
        for param in self.vit_model.parameters():
            param.requires_grad = False
        self.vit_model.eval()

        self.reid_model = FaceReIDModel(embedding_dim=config['visual']['reid_feature_dim'], device=device)
        for param in self.reid_model.parameters():
            param.requires_grad = False
        self.reid_model.eval()

        self.lip_encoder = LipMotionEncoder(
            temporal_window=config['visual']['lip_window_frames'],
            output_dim=config['visual']['lip_feature_dim']
        ).to(device)
        self.lip_encoder.eval()

        self.proj_vit = nn.Sequential(
            nn.Linear(768, config['visual']['proj_vit_dim']),
            nn.LayerNorm(config['visual']['proj_vit_dim'])
        ).to(device)
        self.proj_reid = nn.Sequential(
            nn.Linear(config['visual']['reid_feature_dim'], config['visual']['proj_reid_dim']),
            nn.LayerNorm(config['visual']['proj_reid_dim'])
        ).to(device)
        self.proj_lip = nn.Sequential(
            nn.Linear(config['visual']['lip_feature_dim'], config['visual']['proj_lip_dim']),
            nn.LayerNorm(config['visual']['proj_lip_dim'])
        ).to(device)
        self.proj_vit.eval()
        self.proj_reid.eval()
        self.proj_lip.eval()

        self.lip_buffer = {}
        self.frame_counter = 0

    def extract_features(self, frame_rgb, tracked_faces, timestamp):
        with torch.inference_mode():
            self.frame_counter += 1
            features = {}
            current_track_ids = set()

            for face_info in tracked_faces:
                face_id = face_info['id']
                current_track_ids.add(face_id)
                face_crop = face_info['face_crop']
                if face_crop is None or face_crop.size == 0:
                    continue

                pil_img = Image.fromarray(face_crop)
                vit_inputs = self.vit_processor(images=pil_img, return_tensors="pt").to(self.device)
                vit_emb = self.vit_model(**vit_inputs).last_hidden_state[:, 0, :].squeeze()
                vit_emb = self.proj_vit(vit_emb)

                reid_emb = self.reid_model.extract_feature(face_crop)
                reid_emb = torch.from_numpy(reid_emb).to(self.device)
                reid_emb = self.proj_reid(reid_emb)

                mouth_crop = self._extract_mouth(face_crop)
                if face_id not in self.lip_buffer:
                    self.lip_buffer[face_id] = {'last_frame': self.frame_counter, 'crops': []}
                if self.frame_counter - self.lip_buffer[face_id]['last_frame'] > 2:
                    self.lip_buffer[face_id]['crops'] = []
                self.lip_buffer[face_id]['last_frame'] = self.frame_counter
                self.lip_buffer[face_id]['crops'].append(mouth_crop)
                if len(self.lip_buffer[face_id]['crops']) > self.config['visual']['lip_window_frames']:
                    self.lip_buffer[face_id]['crops'].pop(0)

                lip_emb = torch.zeros(self.config['visual']['proj_lip_dim'], device=self.device)
                if len(self.lip_buffer[face_id]['crops']) == self.config['visual']['lip_window_frames']:
                    mouth_seq = np.stack(self.lip_buffer[face_id]['crops'])
                    mouth_tensor = (
                        torch.from_numpy(mouth_seq)
                        .permute(0, 3, 1, 2)
                        .unsqueeze(0)
                        .float()
                        .to(self.device)
                    )
                    lip_emb = self.lip_encoder(mouth_tensor).squeeze()
                    lip_emb = self.proj_lip(lip_emb)

                full_emb = torch.cat([vit_emb, reid_emb, lip_emb])
                full_emb = torch.nn.functional.normalize(full_emb, p=2, dim=0)
                features[face_id] = full_emb.cpu().numpy()

            for tid in list(self.lip_buffer.keys()):
                if tid not in current_track_ids and self.frame_counter - self.lip_buffer[tid]['last_frame'] > 5:
                    del self.lip_buffer[tid]

            return features

    def _extract_mouth(self, face_crop):
        h, w = face_crop.shape[:2]
        mouth = face_crop[int(h*0.6):h, int(w*0.2):int(w*0.8)]
        mouth = cv2.resize(mouth, (64, 64))
        return mouth
