import numpy as np
import cv2
from facenet_pytorch import MTCNN
import torch
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class FaceTracker:
    def __init__(self, device='cuda', iou_threshold=0.3, max_lost=30, reid_model=None, min_track_length=10):
        self.device = device
        self.detector = MTCNN(keep_all=True, device=device)
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.reid_model = reid_model
        self.next_id = 0
        self.tracks = []
        self.min_track_length = min_track_length

    def update(self, frame_rgb):
        boxes, _ = self.detector.detect(frame_rgb)
        if boxes is None:
            boxes = []
        else:
            boxes = boxes.tolist()

        det_features = []
        if self.reid_model:
            for box in boxes:
                x1,y1,x2,y2 = map(int, box)
                face_crop = frame_rgb[y1:y2, x1:x2]
                if face_crop.size > 0:
                    feat = self.reid_model.extract_feature(face_crop)
                    det_features.append(feat)
                else:
                    det_features.append(None)
        else:
            det_features = [None] * len(boxes)

        for track in self.tracks:
            track['kf'].predict()
            track['lost'] += 1

        matched, unmatched_dets, unmatched_tracks = self._associate(boxes, det_features, frame_rgb)

        for t_idx, d_idx in matched:
            track = self.tracks[t_idx]
            box = boxes[d_idx]
            track['kf'].update(self._box_to_z(box))
            track['lost'] = 0
            track['hits'] += 1
            if self.reid_model and det_features[d_idx] is not None:
                track['features'].append(det_features[d_idx])
                if len(track['features']) > 10:
                    track['features'].pop(0)

        for d_idx in unmatched_dets:
            box = boxes[d_idx]
            kf = self._create_kalman(box)
            features = []
            if self.reid_model and det_features[d_idx] is not None:
                features.append(det_features[d_idx])
            self.tracks.append({'id': self.next_id, 'kf': kf, 'lost': 0, 'features': features, 'hits': 1})
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t['lost'] <= self.max_lost]

        results = []
        for track in self.tracks:
            if track['lost'] == 0 and track['hits'] >= self.min_track_length:
                box = self._z_to_box(track['kf'].x)
                x1,y1,x2,y2 = map(int, box)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame_rgb.shape[1], x2); y2 = min(frame_rgb.shape[0], y2)
                face_crop = frame_rgb[y1:y2, x1:x2] if x2>x1 and y2>y1 else None
                results.append({'id': track['id'], 'bbox': box, 'face_crop': face_crop})
        return results

    def _associate(self, boxes, det_features, frame_rgb):
        if len(self.tracks) == 0 or len(boxes) == 0:
            return [], list(range(len(boxes))), list(range(len(self.tracks)))

        cost_matrix = np.zeros((len(self.tracks), len(boxes)), dtype=np.float32)
        for t_idx, track in enumerate(self.tracks):
            t_box = self._z_to_box(track['kf'].x)
            for d_idx, box in enumerate(boxes):
                iou = self._iou(t_box, box)
                cost = 1 - iou
                if self.reid_model and track['features'] and det_features[d_idx] is not None:
                    reid_dist = min([np.linalg.norm(det_features[d_idx] - f) for f in track['features']])
                    cost += 0.5 * reid_dist
                cost_matrix[t_idx, d_idx] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        all_costs = cost_matrix[row_ind, col_ind]
        thresh = np.percentile(all_costs, 70) if len(all_costs) > 0 else 0.7
        matched = [(r,c) for r,c in zip(row_ind, col_ind) if cost_matrix[r,c] < thresh]
        matched_t = set(r for r,_ in matched)
        matched_d = set(c for _,c in matched)
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_t]
        unmatched_dets = [i for i in range(len(boxes)) if i not in matched_d]
        return matched, unmatched_dets, unmatched_tracks

    def _create_kalman(self, box):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
                         [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        kf.R[2:,2:] *= 10.
        kf.P[4:,4:] *= 1000.
        kf.P *= 10.
        kf.Q[-1,-1] *= 0.01
        kf.Q[4:,4:] *= 0.01
        kf.x[:4] = self._box_to_z(box)
        return kf

    def _box_to_z(self, box):
        return np.array([box[0], box[1], box[2], box[3], 0,0,0]).reshape(7,1)

    def _z_to_box(self, z):
        return z[:4].reshape(4)

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return inter / (areaA + areaB - inter + 1e-6)
