import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import Config

class ImageTransforms:
    def __init__(self, config: Config):
        self.config = config
        
    def square_image_transform(self, image: Image.Image, size: int) -> Image.Image:
        image = image.resize((size, size), Image.ANTIALIAS)
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                self.config.COLOR_JITTER_BRIGHTNESS,
                self.config.COLOR_JITTER_CONTRAST,
                self.config.COLOR_JITTER_SATURATION
            ),
            transforms.RandomCrop((size, size), pad_if_needed=True),
            transforms.RandomAffine(0, translate=None, scale=self.config.SCALE_RANGE)
        ])
        return transform(image)
    
    def rectangle_image_transform(self, image: Image.Image, size: int) -> Image.Image:
        height, width = image.size
        if height > width:
            aspect = width / height
            new_height = int(size)
            new_width = int(np.rint(aspect * size))
        else:
            aspect = height / width
            new_height = int(np.rint(aspect * size))
            new_width = int(size)
        
        image = image.resize((new_height, new_width), Image.ANTIALIAS)
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                self.config.COLOR_JITTER_BRIGHTNESS,
                self.config.COLOR_JITTER_CONTRAST,
                self.config.COLOR_JITTER_SATURATION
            ),
            transforms.RandomCrop((size, size), pad_if_needed=True),
            transforms.RandomAffine(0, translate=None, scale=self.config.SCALE_RANGE)
        ])
        return transform(image)
    
    def apply_transforms(self, image: Image.Image, size: int) -> Image.Image:
        height, width = image.size
        if abs(height - width) < 20:
            return self.square_image_transform(image, size)
        else:
            return self.rectangle_image_transform(image, size)
    
    def normalize_image(self, image: Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.config.IMAGE_MEAN, self.config.IMAGE_STD)
        ])
        return transform(image)

class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, image_path: str, data: pd.DataFrame, size: int, 
                 weight: float = 1.0, noise: bool = False, config: Config = None):
        self.image_path = image_path
        self.size = size
        self.weight = weight
        self.noise = noise
        self.config = config or Config()
        
        data = data.reset_index(drop=True)
        self.image_ids = data['id_code'].tolist()
        self.labels = data['diagnosis'].tolist()
        
        self.transforms = ImageTransforms(self.config)
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _load_image(self, image_id: str) -> torch.Tensor:
        image_path = f"{self.image_path}/{image_id}.jpg"
        try:
            image = Image.open(image_path)
            image = self.transforms.apply_transforms(image, self.size)
            return self.transforms.normalize_image(image)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
    
    def _process_label(self, label_idx: int) -> torch.Tensor:
        label = self.labels[label_idx]
        if self.noise:
            label = np.random.normal(loc=label, scale=self.config.LABEL_NOISE_SCALE)
        
        label_array = np.array([label]).clip(0., 4.)
        return torch.from_numpy(label_array).reshape(-1, 1)
    
    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        image = self._load_image(image_id)
        label = self._process_label(idx)
        weight = torch.from_numpy(np.array([self.weight])).reshape(-1, 1)
        
        return {
            'idx': image_id,
            'image': image,
            'label': label,
            'weight': weight
        }

class NoiseAugmentedDataset(Dataset):
    def __init__(self, image_path: str, data: pd.DataFrame, size: int, 
                 weight: float = 1.0, config: Config = None):
        self.image_path = image_path
        self.size = size
        self.weight = weight
        self.config = config or Config()
        
        data = data.reset_index(drop=True)
        self.image_ids = data['id_code'].tolist()
        self.labels = data['diagnosis'].tolist()
        
        self.transforms = ImageTransforms(self.config)
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _load_image(self, image_id: str) -> torch.Tensor:
        image_path = f"{self.image_path}/{image_id}.jpg"
        try:
            image = Image.open(image_path)
            image = self.transforms.apply_transforms(image, self.size)
            return self.transforms.normalize_image(image)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
    
    def _process_label(self, label_idx: int) -> torch.Tensor:
        label = self.labels[label_idx]
        label = np.random.normal(loc=label, scale=self.config.LABEL_NOISE_SCALE)
        label_array = np.array([label]).clip(0., 4.)
        return torch.from_numpy(label_array).reshape(-1, 1)
    
    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        image_1 = self._load_image(image_id)
        image_2 = self._load_image(image_id)
        label = self._process_label(idx)
        weight = torch.from_numpy(np.array([self.weight])).reshape(-1, 1)
        
        return {
            'idx': image_id,
            'image_1': image_1,
            'image_2': image_2,
            'label': label,
            'weight': weight
        }

def create_data_loaders(image_path: str, label_path: str, size: int, 
                       fold_idx: int, weight: float = 1.0, 
                       use_noise_augmentation: bool = False,
                       config: Config = None) -> tuple:
    config = config or Config()
    
    data = pd.read_csv(label_path)
    train_data = data[data['fold'] != fold_idx].reset_index(drop=True)
    valid_data = data[data['fold'] == fold_idx].reset_index(drop=True)
    
    if use_noise_augmentation:
        train_dataset = NoiseAugmentedDataset(image_path, train_data, size, weight, config)
    else:
        train_dataset = DiabeticRetinopathyDataset(image_path, train_data, size, weight, True, config)
    
    valid_dataset = DiabeticRetinopathyDataset(image_path, valid_data, size, 1.0, False, config)
    
    print(f'Train Images: {len(train_dataset)}, Valid Images: {len(valid_dataset)}')
    
    return train_dataset, valid_dataset
