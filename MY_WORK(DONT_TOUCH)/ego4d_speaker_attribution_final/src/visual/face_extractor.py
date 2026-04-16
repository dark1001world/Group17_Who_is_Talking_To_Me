import cv2
import numpy as np
from facenet_pytorch import MTCNN

class FaceExtractor:
    def __init__(self, face_size=224, device='cuda'):
        self.face_size = face_size
        self.device = device
        self.detector = MTCNN(image_size=face_size, margin=0, keep_all=True, device=device)

    def detect_faces(self, frame_rgb):
        boxes, _ = self.detector.detect(frame_rgb)
        if boxes is None:
            return []
        faces = []
        for box in boxes:
            x1,y1,x2,y2 = map(int, box)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame_rgb.shape[1], x2); y2 = min(frame_rgb.shape[0], y2)
            face_crop = frame_rgb[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, (self.face_size, self.face_size))
                faces.append(face_crop)
        return faces
