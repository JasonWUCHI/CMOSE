import cv2
import os
import random
import pandas as pd
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import av
from torchsampler import ImbalancedDatasetSampler


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

class VideoFolderPathToTensor(object):

    def __init__(self, max_len=None):
        self.max_len = max_len

    def __call__(self, path):
        '''
        path is the folder that stores the frames of a video
        '''

        cap = cv2.VideoCapture(path)
        ret = True
        vframes = []
        while ret:
            ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
            if ret:
                vframes.append(img)
        video = np.stack(vframes, axis=0) # dimensions (T, H, W, C)
        
        height, width, channels = video[0].shape

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            torchvision.transforms.Resize([224, 224], antialias=True)
        ])

        EXTRACT_FREQUENCY = 5

        num_time_steps = 50

        while len(video)<EXTRACT_FREQUENCY*num_time_steps:
            video = np.concatenate((video, video))

        # (3 x T x H x W), https://pytorch.org/docs/stable/torchvision/models.html
        frames = torch.FloatTensor(channels, num_time_steps, 224, 224)

        start_index = random.randint(0,EXTRACT_FREQUENCY-1)
        for index in range(0, num_time_steps):
            frame = torch.from_numpy(video[start_index+index * EXTRACT_FREQUENCY])
            frame = frame.permute(2, 0, 1)  # (H x W x C) to (C x H x W)
            frame = frame / 255
            frame = transform(frame)
            frames[:, index, :, :] = frame.float()

        return frames.permute(1, 0, 2, 3)

class VivitProcessorToTensor(object):

    def __init__(self, max_len=None):
        self.max_len = max_len

    def __call__(self, path):
        
        container = av.open(path)

        if container.streams.video[0].frames <= 160:
            startidx = np.random.randint(0, 4)
            indices = range(startidx,container.streams.video[0].frames,5)
        else:
            indices = sample_frame_indices(clip_len=32, frame_sample_rate=5, seg_len=container.streams.video[0].frames)

        video = read_video_pyav(container=container, indices=indices)

        while len(video)<32:
            video = np.vstack([video, video])
        video = video[:32]

        return list(video)

class VideoDataset(Dataset):

    def __init__(self, data_root, df, transform=None):

        self.transform = transform
        self.data_root = data_root
        self.videolist = list(df['ClipID'])
        self.label = list(df['Engagement'])
 
    def __len__(self):
        return len(self.videolist)

    def __getitem__(self, index):

        video = self.videolist[index]
        label = self.label[index]

        video = os.path.join(self.data_root, video)+".mp4"

        if self.transform != None:
            video_tensor = self.transform(video)
        else:
            video_tensor = video

        return video_tensor, video, label

    def get_labels(self):
        return self.label

def get_dataloader(batch_size, label_df, root, sampler):
    dataloaders = []
    for case in ['Train', 'Validation', 'Test']:
        dataset = VideoDataset(root, label_df[label_df['Split']==case], transform=torchvision.transforms.Compose([VideoFolderPathToTensor()]))
        #dataset = VideoDataset(root, label_df[label_df['Split']==case])
        if case == 'Train' and sampler:
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=ImbalancedDatasetSampler(dataset), num_workers=16)
        else:
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        dataloaders.append(dataloader)

    return dataloaders
