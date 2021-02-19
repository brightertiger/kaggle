import cv2
import torch
import random
import librosa as lb
import numpy as np 
import pandas as pd
import soundfile as sf
import albumentations as A
from torch.utils.data import Dataset
from audiomentations import Compose, AddGaussianNoise, TimeStretch, Shift
from albumentations.pytorch.transforms import ToTensorV2

AUDIO_AUGMENT = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=1.),
], p=1.)

TEST_AUGMENT = A.Compose([
    A.RandomCrop(224, 300, p=1.),
    A.Normalize(p=1.),
    ToTensorV2()
], p=1.)

def specGram(audio):
    params = {}
    params['sr'] = 32000
    params['n_mels'] = 224
    params['fmin'] = 0
    params['fmax'] = None
    spec = lb.feature.melspectrogram(audio, **params)
    spec = lb.power_to_db(spec).astype(np.float32)
    return spec

def colorImage(image, eps=1e-6, mean=None, std=None):
    image = np.stack([image, image, image], axis=-1)
    mean, std = image.mean(), image.std()
    image = (image - mean) / (std + eps)
    _min, _max = image.min(), image.max()
    if (_max - _min) > eps:
        image = np.clip(image, _min, _max)
        image = 255 * (image - _min) / (_max - _min)
        image = image.astype(np.uint8)
    else:
        image = np.zeros_like(image, dtype=np.uint8)
    return image

class ScoreDataset(Dataset):
    
    def __init__(self, data):
        self.path = '/workspace/data/resample/test/'
        self.index = data['recording_id'].tolist()
        self.sr = 32000
        self.length = 5 * self.sr
        self.fmin = 0
        self.fmax = None
        return None
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        name = self.index[index]
        audio = np.load(self.path + name + '.npy')
        segments = len(audio) / self.length
        segments = int(np.ceil(segments)) * 4
        array = []
        start = 0
        for i in range(0, segments):
            begin, end = int(start), int(start + self.length)
            if end > len(audio):
                end = len(audio)
                begin = end - self.length
            start = start + (self.length / 4)
            sliced = audio[begin:end]
            image = specGram(sliced)
            image = colorImage(image)
            image = TEST_AUGMENT(image=image)['image']
            image = image.unsqueeze(0)
            array.append(image)
        array = torch.vstack(array)
        return array
    
def scoreModel(model, data):
    model.eval()
    pred_array = []
    with torch.no_grad():
        for sample in data:
            sound = sample.float().squeeze().to('cuda:0')
            preds = model(sound)
            preds, _ = torch.max(preds, dim=0)
            pred_array.append(preds.cpu().data.numpy())
    pred_array = np.vstack(pred_array)
    return pred_array

