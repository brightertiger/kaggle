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
from typing import Tuple, List
from .config import Config

class AudioProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.audio_augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=1.),
        ], p=1.)
        
        self.train_augment = A.Compose([
            A.RandomCrop(300, 300, p=1.),
            A.CoarseDropout(max_holes=10, p=.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.GaussNoise(p=.5),
            A.Normalize(p=1.),
            ToTensorV2()
        ], p=1.)
        
        self.valid_augment = A.Compose([
            A.RandomCrop(300, 300, p=1.),
            A.Normalize(p=1.),
            ToTensorV2()
        ], p=1.)
        
        self.test_augment = A.Compose([
            A.RandomCrop(300, 300, p=1.),
            A.Normalize(p=1.),
            ToTensorV2()
        ], p=1.)

    def create_melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        params = {
            'sr': self.config.audio.sample_rate,
            'n_mels': self.config.audio.n_mels,
            'fmin': self.config.audio.fmin,
            'fmax': self.config.audio.fmax
        }
        spec = lb.feature.melspectrogram(audio, **params)
        spec = lb.power_to_db(spec).astype(np.float32)
        return spec

    def convert_to_rgb(self, image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
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

    def extract_audio_segment(self, audio: np.ndarray, t_min: float, t_max: float, 
                            apply_augmentation: bool = False) -> np.ndarray:
        t_min_samples = float(t_min) * self.config.audio.sample_rate
        t_max_samples = float(t_max) * self.config.audio.sample_rate
        center = np.round((t_min_samples + t_max_samples) / 2) + np.random.uniform(-1, 1)
        
        length_samples = self.config.audio.segment_length * self.config.audio.sample_rate
        beginning = max(center - length_samples / 2, 0)
        ending = beginning + length_samples
        
        if ending > len(audio):
            ending = len(audio)
            beginning = ending - length_samples
            
        audio_segment = audio[int(beginning):int(ending)]
        
        if apply_augmentation:
            audio_segment = self.audio_augment(samples=audio_segment, 
                                            sample_rate=self.config.audio.sample_rate)
        
        return audio_segment

class TrainDataset(Dataset):
    def __init__(self, data: pd.DataFrame, fold: int, config: Config):
        self.data = data[data['fold'] != fold].reset_index(drop=True)
        self.config = config
        self.processor = AudioProcessor(config)
        self.index = list(range(len(self.data)))

    def __len__(self) -> int:
        return len(self.index)

    def _get_labels(self, idx: int) -> Tuple[int, float, float]:
        row = self.data.iloc[idx]
        return int(row['species_id']), float(row['t_min']), float(row['t_max'])

    def _create_image(self, audio: np.ndarray, t_min: float, t_max: float, 
                     apply_augmentation: bool = True) -> torch.Tensor:
        audio_segment = self.processor.extract_audio_segment(audio, t_min, t_max, apply_augmentation)
        spec = self.processor.create_melspectrogram(audio_segment)
        image = self.processor.convert_to_rgb(spec)
        
        if apply_augmentation:
            image = self.processor.train_augment(image=image)['image']
        else:
            image = self.processor.valid_augment(image=image)['image']
            
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx1 = random.choice(self.index)
        idx2 = random.choice(self.index)
        lam = np.random.uniform(0.8, 1.0)
        
        name1 = self.data.iloc[idx1]['recording_id']
        name2 = self.data.iloc[idx2]['recording_id']
        
        audio1 = np.load(f"{self.config.data.audio_data_path}/train/{name1}.npy")
        audio2 = np.load(f"{self.config.data.audio_data_path}/train/{name2}.npy")
        
        label1, tmin1, tmax1 = self._get_labels(idx1)
        label2, tmin2, tmax2 = self._get_labels(idx2)
        
        image1 = self._create_image(audio1, tmin1, tmax1, apply_augmentation=True)
        image2 = self._create_image(audio2, tmin2, tmax2, apply_augmentation=True)
        
        image = lam * image1 + (1. - lam) * image2
        
        label1_vec = np.zeros(self.config.model.num_classes)
        label2_vec = np.zeros(self.config.model.num_classes)
        label1_vec[label1] = 1.0
        label2_vec[label2] = 1.0
        
        label = lam * label1_vec + (1. - lam) * label2_vec
        
        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.float)

class ValidDataset(Dataset):
    def __init__(self, data: pd.DataFrame, fold: int, config: Config):
        self.data = data[data['fold'] == fold].reset_index(drop=True)
        self.config = config
        self.processor = AudioProcessor(config)

    def __len__(self) -> int:
        return len(self.data)

    def _get_labels(self, idx: int) -> Tuple[int, float, float]:
        row = self.data.iloc[idx]
        return int(row['species_id']), float(row['t_min']), float(row['t_max'])

    def _create_image(self, audio: np.ndarray, t_min: float, t_max: float) -> torch.Tensor:
        audio_segment = self.processor.extract_audio_segment(audio, t_min, t_max, apply_augmentation=False)
        spec = self.processor.create_melspectrogram(audio_segment)
        image = self.processor.convert_to_rgb(spec)
        image = self.processor.valid_augment(image=image)['image']
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        name = row['recording_id']
        audio = np.load(f"{self.config.data.audio_data_path}/train/{name}.npy")
        
        label, tmin, tmax = self._get_labels(idx)
        image = self._create_image(audio, tmin, tmax)
        
        label_vec = np.zeros(self.config.model.num_classes)
        label_vec[label] = 1.0
        
        return torch.tensor(image, dtype=torch.float), torch.tensor(label_vec, dtype=torch.float)

class TestDataset(Dataset):
    def __init__(self, data: pd.DataFrame, config: Config, apply_tta: bool = False):
        self.data = data
        self.config = config
        self.processor = AudioProcessor(config)
        self.apply_tta = apply_tta

    def __len__(self) -> int:
        return len(self.data)

    def _create_image(self, audio: np.ndarray, apply_augmentation: bool = False) -> torch.Tensor:
        if apply_augmentation:
            audio = self.processor.audio_augment(samples=audio, 
                                              sample_rate=self.config.audio.sample_rate)
        
        spec = self.processor.create_melspectrogram(audio)
        image = self.processor.convert_to_rgb(spec)
        
        if apply_augmentation:
            image = self.processor.train_augment(image=image)['image']
        else:
            image = self.processor.test_augment(image=image)['image']
            
        return image

    def __getitem__(self, idx: int) -> torch.Tensor:
        name = self.data.iloc[idx]['recording_id']
        audio = np.load(f"{self.config.data.audio_data_path}/test/{name}.npy")
        
        length_samples = self.config.audio.segment_length * self.config.audio.sample_rate
        segments = int(np.ceil(len(audio) / length_samples)) * 4
        
        images = []
        start = 0
        
        for i in range(segments):
            begin = int(start)
            end = int(start + length_samples)
            
            if end > len(audio):
                end = len(audio)
                begin = end - length_samples
                
            start += length_samples / 4
            audio_segment = audio[begin:end]
            
            image = self._create_image(audio_segment, apply_augmentation=self.apply_tta)
            images.append(image.unsqueeze(0))
        
        return torch.vstack(images)

def create_data_loaders(config: Config, fold: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    data = pd.read_csv(config.data.train_data_path)
    
    train_dataset = TrainDataset(data, fold, config)
    valid_dataset = ValidDataset(data, fold, config)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f'Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}')
    return train_loader, valid_loader

def create_test_loader(config: Config, apply_tta: bool = False) -> torch.utils.data.DataLoader:
    data = pd.read_csv(config.data.test_data_path)
    test_dataset = TestDataset(data, config, apply_tta=apply_tta)
    
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.training.num_workers
    )
