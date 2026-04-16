import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
import numpy as np
from PIL import Image

class FaceReIDModel(nn.Module):
    def __init__(self, embedding_dim=512, pretrained='vggface2', device='cuda'):
        super().__init__()
        self.device = device
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False).to(device)
        self.backbone.eval()
        self.projection = nn.Linear(512, embedding_dim).to(device)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, face_crops):
        features = self.backbone(face_crops)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)

    def extract_feature(self, face_crop_np):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160,160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        img_tensor = transform(face_crop_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.forward(img_tensor).squeeze().cpu().numpy()
        return emb
