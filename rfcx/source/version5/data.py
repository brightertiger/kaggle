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

TRAIN_AUGMENT = A.Compose([
    A.RandomCrop(224, 300, p=1.),
    A.CoarseDropout(max_holes=10, p=.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    A.GaussNoise(p=.5),
    A.Normalize(p=1.),
    ToTensorV2()
], p=1.)

VALID_AUGMENT = A.Compose([
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

class TrainDataset(Dataset):

    def __init__(self, data, fold):
        self.period = 5
        self.data = data
        self.path = '/workspace/data/raw/train/'
        self.data = self.data[self.data['fold'] != fold]
        self.data = self.data.reset_index(drop=True).reset_index()
        self.index = self.data['index'].tolist()
        self.sr = 32000
        self.length = 5 * self.sr
        self.fmin = 0
        self.fmax = None
        return None
    
    def __len__(self):
        return len(self.index)
    
    def __labels__(self, idx):
        data = self.data[self.data['index'] == idx]
        label = data['species_id'].tolist()[0]
        tmin = data['t_min'].tolist()[0]
        tmax = data['t_max'].tolist()[0]
        return label, tmin, tmax
    
    def __image__(self, audio, t_min, t_max):
        t_min = float(t_min) * self.sr
        t_max = float(t_max) * self.sr
        center = np.round((t_min + t_max) / 2) + np.random.uniform(-1,1)
        beginning = max(center - self.length / 2, 0) 
        ending = beginning + self.length
        if ending > len(audio):
            ending = len(audio)
            beginning = ending - self.length
        audio = audio[int(beginning):int(ending)]
        audio = AUDIO_AUGMENT(samples=audio, sample_rate=self.sr)
        image = specGram(audio)
        image = colorImage(image)
        image = TRAIN_AUGMENT(image=image)['image']
        return image

    def __getitem__(self, idx):
        idx = random.choice(self.index)
        name = self.data.at[idx, 'recording_id']
        audio = np.load('../../data/resample/train/{}.npy'.format(name))
        label, tmin, tmax = self.__labels__(idx)
        label_ = [0] * 24
        label_[label] += 1 
        label_ = np.array(label_).astype(int)
        image = self.__image__(audio, tmin, tmax)
        image = torch.tensor(image, dtype=torch.float)
        label = torch.tensor(label_, dtype=torch.long).reshape(24,)
        sample = (image, label)
        return sample
    
class ValidDataset(Dataset):

    def __init__(self, data, fold):
        self.period = 5
        self.data = data
        self.path = '/workspace/data/raw/train/'
        self.data = self.data[self.data['fold'] == fold]
        self.data = self.data.reset_index(drop=True).reset_index()
        self.index = self.data['index'].tolist()
        self.sr = 32000
        self.length = 5 * self.sr
        self.fmin = 0
        self.fmax = None
        return None
    
    def __len__(self):
        return len(self.index)
    
    def __labels__(self, idx):
        data = self.data[self.data['index'] == idx]
        label = data['species_id'].tolist()[0]
        tmin = data['t_min'].tolist()[0]
        tmax = data['t_max'].tolist()[0]
        return label, tmin, tmax
    
    def __image__(self, audio, t_min, t_max):
        t_min = float(t_min) * self.sr
        t_max = float(t_max) * self.sr
        center = np.round((t_min + t_max) / 2) + np.random.uniform(-1,1)
        beginning = max(center - self.length / 2, 0)
        ending = beginning + self.length
        if ending > len(audio):
            ending = len(audio)
            beginning = ending - self.length
        audio = audio[int(beginning):int(ending)]
        image = specGram(audio)
        image = colorImage(image)
        image = VALID_AUGMENT(image=image)['image']
        return image

    def __getitem__(self, idx):
        idx = random.choice(self.index)
        name = self.data.at[idx, 'recording_id']
        audio = np.load('../../data/resample/train/{}.npy'.format(name))
        label, tmin, tmax = self.__labels__(idx)
        label_ = [0] * 24
        label_[label] += 1 
        label_ = np.array(label_).astype(int)
        image = self.__image__(audio, tmin, tmax)
        image = torch.tensor(image, dtype=torch.float)
        label = torch.tensor(label_, dtype=torch.long).reshape(24,)
        sample = (image, label)
        return sample
    
def trainLoader(data, fold_idx):
    data = pd.read_csv(data)
    train = TrainDataset(data, fold_idx)
    valid = ValidDataset(data, fold_idx)
    print('Train Images:', len(train), 'Valid Images:', len(valid))
    return train, valid