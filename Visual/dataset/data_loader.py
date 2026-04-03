import os, cv2, json, glob, logging
import torch
import hashlib
import torchvision.transforms as transforms
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict, OrderedDict

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def helper():
    return defaultdict(OrderedDict)


def pad_video(video):
    assert len(video) == 7
    pad_idx = np.all(video == 0, axis=(1, 2, 3))
    mid_idx = int(len(pad_idx) / 2)
    pad_idx[mid_idx] = False
    pad_frames = video[~pad_idx]
    pad_frames = np.pad(pad_frames, ((sum(pad_idx[:mid_idx]), 0), (0, 0), (0, 0), (0, 0)), mode='edge')
    pad_frames = np.pad(pad_frames, ((0, sum(pad_idx[mid_idx + 1:])), (0, 0), (0, 0), (0, 0)), mode='edge')
    return pad_frames.astype(np.uint8)


def check(track):
    inter_track = []
    framenum = []
    bboxes = []
    for frame in track:
        x = frame['x']
        y = frame['y']
        w = frame['width']
        h = frame['height']
        if (w <= 0 or h <= 0 or
                frame['frameNumber'] == 0 or
                len(frame['Person ID']) == 0):
            continue
        framenum.append(frame['frameNumber'])
        x = max(x, 0)
        y = max(y, 0)
        bbox = [x, y, x + w, y + h]
        bboxes.append(bbox)

    if len(framenum) == 0:
        return inter_track

    framenum = np.array(framenum)
    bboxes = np.array(bboxes)

    gt_frames = framenum[-1] - framenum[0] + 1

    frame_i = np.arange(framenum[0], framenum[-1] + 1)

    if gt_frames > framenum.shape[0]:
        bboxes_i = []
        for ij in range(0, 4):
            interpfn = interp1d(framenum, bboxes[:, ij])
            bboxes_i.append(interpfn(frame_i))
        bboxes_i = np.stack(bboxes_i, axis=1)
    else:
        frame_i = framenum
        bboxes_i = bboxes

    # assemble new tracklet
    template = track[0]
    for i, (frame, bbox) in enumerate(zip(frame_i, bboxes_i)):
        record = template.copy()
        record['frameNumber'] = frame
        record['x'] = bbox[0]
        record['y'] = bbox[1]
        record['width'] = bbox[2] - bbox[0]
        record['height'] = bbox[3] - bbox[1]
        inter_track.append(record)
    return inter_track


def make_dataset(file_name, json_path, gt_path, stride=1):
    logger.info('load: ' + file_name)

    images = []
    keyframes = []
    count = 0

    with open(file_name, 'r') as f:
        videos = f.readlines()
    for uid in videos:
        uid = uid.strip()

        with open(os.path.join(gt_path, uid + '.json')) as f:
            gts = json.load(f)
        positive = set()
        for gt in gts:
            for i in range(gt['start_frame'], gt['end_frame'] + 1):
                positive.add(str(i) + ":" + gt['label'])

        vid_json_dir = os.path.join(json_path, uid)
        tracklets = glob.glob(f'{vid_json_dir}/*.json')
        for t in tracklets:
            with open(t, 'r') as j:
                frames = json.load(j)
            frames.sort(key=lambda x: x['frameNumber'])
            trackid = os.path.basename(t)[:-5]
            # check the bbox, interpolate when necessary
            frames = check(frames)

            for idx, frame in enumerate(frames):
                frameid = frame['frameNumber']
                bbox = (frame['x'],
                        frame['y'],
                        frame['x'] + frame['width'],
                        frame['y'] + frame['height'])
                identifier = str(frameid) + ':' + frame['Person ID']
                label = 1 if identifier in positive else 0
                images.append((uid, trackid, frameid, bbox, label))
                if idx % stride == 0:
                    keyframes.append(count)
                count += 1

    return images, keyframes


def make_test_dataset(test_path, stride=1):
    logger.info('load: ' + test_path)

    g = os.walk(test_path)
    images = []
    keyframes = []
    count = 0

    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            if os.path.exists(os.path.join(test_path, dir_name)):
                uid = dir_name
                g2 = os.walk(os.path.join(test_path, uid))
                for _, track_list, _ in g2:
                    for track_id in track_list:
                        g3 = os.walk(os.path.join(test_path, uid, track_id))
                        for _, _, frame_list in g3:
                            for idx, frames in enumerate(frame_list):
                                frame_id = frames.split('_')[0]
                                unique_id = frames.split('_')[1].split('.')[0]
                                images.append((uid, track_id, unique_id, frame_id))
                                if idx % stride == 0:
                                    keyframes.append(count)
                                count += 1
    return images, keyframes


class ImagerLoader(torch.utils.data.Dataset):
    def __init__(self, source_path, file_name, json_path, gt_path,
                 stride=1, scale=0, mode='train', transform=None):

        self.source_path = source_path
        assert os.path.exists(self.source_path), 'source path not exist'
        self.file_name = file_name
        assert os.path.exists(self.file_name), f'{mode} list not exist'
        self.json_path = json_path
        assert os.path.exists(self.json_path), 'json path not exist'

        images, keyframes = make_dataset(file_name, json_path, gt_path, stride=stride)
        self.imgs = images
        self.kframes = keyframes
        self.img_group = self._get_img_group()
        self.scale = scale  # box expand ratio
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        source_video = self._get_video(index)
        target = self._get_target(index)
        return source_video, target

    def __len__(self):
        return len(self.kframes)

    def _get_video(self, index, debug=False):
        uid, trackid, frameid, _, label = self.imgs[self.kframes[index]]
        video = []
        need_pad = False
        for i in range(frameid - 3, frameid + 4):

            img = f'{self.source_path}/{uid}/img_{i:05d}.jpg'
            if i not in self.img_group[uid][trackid] or not os.path.exists(img):
                video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                if not need_pad:
                    need_pad = True
                continue

            assert os.path.exists(img), f'img: {img} not found'
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            bbox = self.img_group[uid][trackid][i]
            x1 = int((1.0 - self.scale) * bbox[0])
            y1 = int((1.0 - self.scale) * bbox[1])
            x2 = int((1.0 + self.scale) * bbox[2])
            y2 = int((1.0 + self.scale) * bbox[3])
            face = img[y1: y2, x1: x2, :]
            try:
                face = cv2.resize(face, (224, 224))
            except:
                # bad bbox
                print('bad bbox, pad with zero')
                face = np.zeros((224, 224, 3), dtype=np.uint8)

            if debug:
                import matplotlib.pyplot as plt
                plt.imshow(face)
                plt.show()

            video.append(np.expand_dims(face, axis=0))

        video = np.concatenate(video, axis=0)
        if need_pad:
            video = pad_video(video)

        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)

        return video.type(torch.float32)

    def _get_target(self, index):
        if self.mode == 'train':
            label = self.imgs[self.kframes[index]][-1]
            return torch.LongTensor([label])
        else:
            return self.imgs[self.kframes[index]]

    def _get_img_group(self):
        img_group = self._nested_dict()
        for db in self.imgs:
            img_group[db[0]][db[1]][db[2]] = db[3]
        return img_group

    def _nested_dict(self):
        return defaultdict(helper)


class TestImagerLoader(torch.utils.data.Dataset):
    def __init__(self, test_path, stride=1, transform=None):

        self.test_path = test_path
        assert os.path.exists(self.test_path), 'test dataset path not exist'

        images, keyframes = make_test_dataset(test_path, stride=stride)
        self.imgs = images
        self.kframes = keyframes
        self.transform = transform

    def __getitem__(self, index):
        source_video = self._get_video(index)
        target = self._get_target(index)
        return source_video, target

    def __len__(self):
        return len(self.kframes)

    def _get_video(self, index):
        uid, trackid, uniqueid, frameid = self.imgs[self.kframes[index]]
        video = []
        need_pad = False

        path = os.path.join(self.test_path, uid, trackid)
        for i in range(int(frameid) - 3, int(frameid) + 4):
            found = False
            ii = str(i).zfill(5)
            g = os.walk(path)
            for _, _, file_list in g:
                for f in file_list:
                    
                    if ii in f:
                        img_path = os.path.join(path, f)
                        img = cv2.imread(img_path)
                     
                        if np.shape(img) == (): 
                            continue
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        video.append(np.expand_dims(img, axis=0))
                        found = True
                        break
                        
                if not found:
                    video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                    if not need_pad:
                        need_pad = True

        video = np.concatenate(video, axis=0)
        if need_pad:
            video = pad_video(video)

        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)

        return video.type(torch.float32)

    def _get_target(self, index):
        return self.imgs[self.kframes[index]]


def make_infer_dataset(test_path, stride=1):
    logger.info('load: ' + test_path)

    g = os.walk(test_path)
    images = []
    keyframes = []
    count = 0
    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            if os.path.exists(os.path.join(test_path, dir_name)):
                uid = dir_name
                g2 = os.walk(os.path.join(test_path, uid) + '/face')
                for _, _, track_list in g2:
                    t = []
                    for i in range(len(track_list)):
                        track_list[i] = (track_list[i].split('.')[0])
                        if track_list[i].isdigit(): 
                            t.append(int(track_list[i]))

                    t = sorted(t)
                   
                    for track_id in t:
                        frame_id = track_id
             
                        images.append((uid, frame_id))
                        if frame_id % stride == 0:
                            keyframes.append(frame_id)
                        count += 1
                      
    return images, keyframes

class InferImagerLoader(torch.utils.data.Dataset):
    def __init__(self, test_path, stride=1, transform=None):

        self.test_path = test_path
        assert os.path.exists(self.test_path), 'test dataset path not exist'

        images, keyframes = make_infer_dataset(test_path, stride=stride)
        self.imgs = images
        self.kframes = keyframes
        self.transform = transform

    def __getitem__(self, index):
        
        source_video = self._get_video(index)
        target = self._get_target(index)
     
        return source_video, target

    def __len__(self):
        return len(self.kframes)

    def _get_video(self, index):
        uid, frameid = self.imgs[index]
        video = []
        need_pad = False

        path = os.path.join(self.test_path, uid)
                         
        for i in range(int(frameid) - 3, int(frameid) + 4):
            found = False
            g = os.walk(path)
            i = str(i) + '.jpg'
          
            for _, _, file_list in g:

                if i in file_list:
                    img_path = os.path.join(path, i)
                    img = cv2.imread(img_path)
                 
                    if np.shape(img) == (): 
                        continues
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    video.append(np.expand_dims(img, axis=0))
                    found = True

                    break
                        
                if not found:
                    video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                    if not need_pad:
                        need_pad = True
                break
                
        
        video = np.concatenate(video, axis=0)
        if need_pad:
            video = pad_video(video)

        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)
        return video.type(torch.float32)

    def _get_target(self, index):
        return self.imgs[index]

## Load Pre Cropped Face image
    
def make_face_dataset(file_name, json_path, gt_path, stride=1):
    logger.info('load: ' + file_name)

    images = []
    keyframes = []
    count = 0
        
    with open(file_name, 'r') as f:
        videos = f.readlines()
    for uid in videos:
        uid = uid.strip()

        with open(os.path.join(gt_path, uid + '.json')) as f:
            gts = json.load(f)
        positive = set()
        for gt in gts:
            for i in range(gt['start_frame'], gt['end_frame'] + 1):
                positive.add(str(i) + ":" + gt['label'])

        vid_json_dir = os.path.join(json_path, uid)
        tracklets = glob.glob(f'{vid_json_dir}/*.json')
        for t in tracklets:
            with open(t, 'r') as j:
                frames = json.load(j)
            frames.sort(key=lambda x: x['frameNumber'])
            trackid = os.path.basename(t)[:-5]
            # check the bbox, interpolate when necessary
            frames = check(frames)

            for idx, frame in enumerate(frames):
                frameid = frame['frameNumber']
                bbox = (frame['x'],
                        frame['y'],
                        frame['x'] + frame['width'],
                        frame['y'] + frame['height'])
                identifier = str(frameid) + ':' + frame['Person ID']
                label = 1 if identifier in positive else 0
                images.append((uid, trackid, frameid, bbox, label))
                if idx % stride == 0:
                    keyframes.append(count)
                count += 1

    return images, keyframes

class FaceImagerLoader(torch.utils.data.Dataset):
    def __init__(self, face_path, file_name, json_path, gt_path,
                 stride=1, scale=0, mode='train', transform=None):

        self.face_path = face_path
        assert os.path.exists(self.face_path), 'face path not exist'
        self.file_name = file_name
        assert os.path.exists(self.file_name), f'{mode} list not exist'
        self.json_path = json_path
        assert os.path.exists(self.json_path), 'json path not exist'

        images, keyframes = make_face_dataset(file_name, json_path, gt_path, stride=stride)
        self.imgs = images
        self.kframes = keyframes
        self.img_group = self._get_img_group()
        self.scale = scale  # box expand ratio
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        source_video = self._get_video(index)
        target = self._get_target(index)
        return source_video, target

    def __len__(self):
        return len(self.kframes)

    def _get_video(self, index, debug=False):
        uid, trackid, frameid, _, label = self.imgs[self.kframes[index]]
        video = []
        need_pad = False
        key=uid+":"+trackid+":"+str(frameid)  
        
        for i in range(frameid - 3, frameid + 4):
            uid_path=os.path.join(self.face_path,uid)
            track_path=os.path.join(uid_path,trackid)
            face_path=f'{track_path}/face_{i:05d}.jpg'
                        
            
            if i not in self.img_group[uid][trackid] or not os.path.exists(face_path):
                video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                if not need_pad:
                    need_pad = True
                continue

            assert os.path.exists(face_path), f'img: {face_img} not found'
            face_img = cv2.imread(face_path)
            face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
         
            try:
                face = cv2.resize(face, (224, 224))
            except:
                # bad bbox
                print('bad bbox, pad with zero')
                face = np.zeros((224, 224, 3), dtype=np.uint8)

            if debug:
                import matplotlib.pyplot as plt
                plt.imshow(face)
                plt.show()

            video.append(np.expand_dims(face, axis=0))

        video = np.concatenate(video, axis=0)
        if need_pad:
            video = pad_video(video)

        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)
            # video = video.view(21,224,224)

        return video.type(torch.float32)

    def _get_target(self, index):
        if self.mode == 'train':
            label = self.imgs[self.kframes[index]][-1]
            return torch.LongTensor([label])
        else:
            return self.imgs[self.kframes[index]]

    def _get_img_group(self):
        img_group = self._nested_dict()
        for db in self.imgs:
            img_group[db[0]][db[1]][db[2]] = db[3]
        return img_group

    def _nested_dict(self):
        return defaultdict(helper)
    
    
    
## Load Pre Cropped Face image and Face Landmark image
    
def make_face_landmark_dataset(file_name, json_path, gt_path, stride=1):
    logger.info('load: ' + file_name)

    images = []
    keyframes = []
    count = 0
        
    with open(file_name, 'r') as f:
        videos = f.readlines()
    for uid in videos:
        uid = uid.strip()

        with open(os.path.join(gt_path, uid + '.json')) as f:
            gts = json.load(f)
        positive = set()
        for gt in gts:
            for i in range(gt['start_frame'], gt['end_frame'] + 1):
                positive.add(str(i) + ":" + gt['label'])

        vid_json_dir = os.path.join(json_path, uid)
        tracklets = glob.glob(f'{vid_json_dir}/*.json')
        for t in tracklets:
            with open(t, 'r') as j:
                frames = json.load(j)
            frames.sort(key=lambda x: x['frameNumber'])
            trackid = os.path.basename(t)[:-5]
            # check the bbox, interpolate when necessary
            frames = check(frames)

            for idx, frame in enumerate(frames):
                frameid = frame['frameNumber']
                bbox = (frame['x'],
                        frame['y'],
                        frame['x'] + frame['width'],
                        frame['y'] + frame['height'])
                identifier = str(frameid) + ':' + frame['Person ID']
                label = 1 if identifier in positive else 0
                images.append((uid, trackid, frameid, bbox, label))
                if idx % stride == 0:
                    keyframes.append(count)
                count += 1

    return images, keyframes


from msgspec.json import decode
from msgspec import Struct

class Repo(Struct):
    name: str

class Actor(Struct):
    login: str

class Interaction(Struct):
    actor: Actor
    repo: Repo

                    
class FaceLandmarkImagerLoader(torch.utils.data.Dataset):
    def __init__(self, face_path, face_landmark_path, file_name, json_path, gt_path,
                 stride=1, scale=0, mode='train', transform=None):

        self.face_path = face_path
        assert os.path.exists(self.face_path), 'face path not exist'
        self.face_landmark_path = face_landmark_path
        assert os.path.exists(self.face_landmark_path), 'face landmark path not exist'
        self.file_name = file_name
        assert os.path.exists(self.file_name), f'{mode} list not exist'
        self.json_path = json_path
        assert os.path.exists(self.json_path), 'json path not exist'
        
        folder_dir = face_landmark_path
        video = os.listdir(folder_dir)
        video.sort()
        pose = dict()
        for video_file in video:
            clip = os.listdir(folder_dir + '/' + video_file)
            clip.sort()
            for track in clip:
                split_tup = os.path.splitext(track)
                if split_tup[1] != '.json':
                    continue
                file_path = (folder_dir + '/' + video_file + '/' + track)
                key = video_file + '/' + track
                with open(file_path, "rb") as f:
                    data = decode(f.read())
                for element in data['instance_info']:
                    landmark = element['instances'][0]['keypoints']
                    score = element['instances'][0]['keypoint_scores']

                    key = video_file + '/' + track + '/' + element['frame_name']
                    pose[key] = np.column_stack((landmark, score))
            
        self.mmpose = pose

        images, keyframes = make_face_landmark_dataset(file_name, json_path, gt_path, stride=stride)
        self.imgs = images
        self.kframes = keyframes
        self.img_group = self._get_img_group()
        self.scale = scale  # box expand ratio
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        source_video, face_landmark = self._get_video(index)
        target = self._get_target(index)
        
        return source_video, target, face_landmark

    def __len__(self):
        return len(self.kframes)

    def _get_video(self, index, debug=False):
        uid, trackid, frameid, _, label = self.imgs[self.kframes[index]]
        video = []
        
        need_pad = False
        
        ## Load face landmark 
        face_landmark_video = []
        landmark_key = uid + '/' + trackid + '/' + f'face_{frameid:05d}.jpg'

        if landmark_key in self.mmpose.keys():
            face_landmark_video.append(np.expand_dims(self.mmpose[key], axis=(0)))
        else:
            face_landmark_video.append(np.zeros((1, 106, 3), dtype=np.float32))
                
        for i in range(frameid - 3, frameid + 4):
            uid_path=os.path.join(self.face_path,uid)
            track_path=os.path.join(uid_path,trackid)
            face_path=f'{track_path}/face_{i:05d}.jpg'
            
            if i not in self.img_group[uid][trackid] or not os.path.exists(face_path):
                video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                if not need_pad:
                    need_pad = True
                continue

            assert os.path.exists(face_path), f'img: {face_img} not found'
            face_img = cv2.imread(face_path)
            face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
         
            try:
                face = cv2.resize(face, (224, 224))
            except:
                # bad bbox
                print('bad bbox, pad with zero')
                face = np.zeros((224, 224, 3), dtype=np.uint8)

            if debug:
                import matplotlib.pyplot as plt
                plt.imshow(face)
                plt.show()

            video.append(np.expand_dims(face, axis=0))
            
        
        
        video = np.concatenate(video, axis=0)
        
        if need_pad:
            video = pad_video(video)

        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)

        face_landmark_video = np.concatenate(face_landmark_video, axis=0)
        face_landmark_video = np.clip(face_landmark_video, 0, 224)
        face_landmark_video = torch.Tensor(face_landmark_video)

        return video.type(torch.float32), face_landmark_video.type(torch.float32)

    def _get_target(self, index):
        if self.mode == 'train':
            label = self.imgs[self.kframes[index]][-1]
            return torch.LongTensor([label])
        else:
            return self.imgs[self.kframes[index]]

    def _get_img_group(self):
        img_group = self._nested_dict()
        for db in self.imgs:
            img_group[db[0]][db[1]][db[2]] = db[3]
        return img_group

    def _nested_dict(self):
        return defaultdict(helper)
