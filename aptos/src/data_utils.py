"""RGB loading, retinal image augmentation and fold datasets."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import Config


class ImageTransforms:
    def __init__(self, config: Config):
        self.config = config

    def _augment(self, image: Image.Image, size: int) -> Image.Image:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                self.config.COLOR_JITTER_BRIGHTNESS,
                self.config.COLOR_JITTER_CONTRAST,
                self.config.COLOR_JITTER_SATURATION,
            ),
            transforms.RandomCrop((size, size), pad_if_needed=True),
            transforms.RandomAffine(0, scale=self.config.SCALE_RANGE),
        ])(image)

    def _resize(self, image: Image.Image, size: int) -> Image.Image:
        width, height = image.size
        if abs(height - width) < 20:
            shape = (size, size)
        else:
            scale = size / max(width, height)
            shape = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(shape, Image.Resampling.LANCZOS)

    def square_image_transform(self, image: Image.Image, size: int) -> Image.Image:
        return self._augment(image.resize((size, size), Image.Resampling.LANCZOS), size)

    def rectangle_image_transform(self, image: Image.Image, size: int) -> Image.Image:
        return self._augment(self._resize(image, size), size)

    def apply_transforms(self, image: Image.Image, size: int, augment: bool = True) -> Image.Image:
        image = self._resize(image, size)
        if augment:
            return self._augment(image, size)
        # CenterCrop pads smaller images with black before cropping.
        return transforms.CenterCrop((size, size))(image)

    def normalize_image(self, image: Image.Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.config.IMAGE_MEAN, self.config.IMAGE_STD),
        ])(image)


class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, image_path: str, data: pd.DataFrame, size: int,
                 weight: float = 1.0, noise: bool = False, config: Config = None,
                 augment: bool | None = None):
        self.image_path = Path(image_path)
        self.size = size
        self.weight = weight
        self.noise = noise
        self.augment = noise if augment is None else augment
        self.config = config or Config()
        self.image_ids = data['id_code'].astype(str).tolist()
        self.labels = data['diagnosis'].tolist() if 'diagnosis' in data else None
        self.transforms = ImageTransforms(self.config)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, image_id: str) -> torch.Tensor:
        # Native APTOS uses PNG; the migrated external dataset used JPEG.
        candidates = [self.image_path / f'{image_id}{ext}' for ext in ('.png', '.jpg', '.jpeg')]
        image_path = next((path for path in candidates if path.is_file()), None)
        if image_path is None:
            raise FileNotFoundError(f'No PNG/JPEG image for {image_id!r} in {self.image_path}')
        with Image.open(image_path) as source:
            image = self.transforms.apply_transforms(source.convert('RGB'), self.size, self.augment)
        return self.transforms.normalize_image(image)

    def _process_label(self, idx: int) -> torch.Tensor:
        label = self.labels[idx]
        if self.noise:
            label = np.random.normal(label, self.config.LABEL_NOISE_SCALE)
        return torch.tensor(float(np.clip(label, 0., 4.)), dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        item = {'idx': self.image_ids[idx], 'image': self._load_image(self.image_ids[idx])}
        if self.labels is not None:
            item.update(label=self._process_label(idx), weight=torch.tensor(self.weight, dtype=torch.float32))
        return item


class NoiseAugmentedDataset(DiabeticRetinopathyDataset):
    def __init__(self, image_path: str, data: pd.DataFrame, size: int,
                 weight: float = 1.0, config: Config = None):
        super().__init__(image_path, data, size, weight, True, config, augment=True)

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item['image_1'] = item.pop('image')
        item['image_2'] = self._load_image(self.image_ids[idx])
        return item


def create_data_loaders(image_path: str, label_path: str, size: int,
                        fold_idx: int, weight: float = 1.0,
                        use_noise_augmentation: bool = False,
                        config: Config = None) -> tuple:
    """Return training/validation datasets (historical function name retained)."""
    config = config or Config()
    data = pd.read_csv(label_path, dtype={'id_code': str})
    if fold_idx not in data['fold'].unique():
        raise ValueError(f'Fold {fold_idx} is absent from {label_path}')
    train_data = data[data['fold'] != fold_idx].reset_index(drop=True)
    valid_data = data[data['fold'] == fold_idx].reset_index(drop=True)
    if train_data.empty or valid_data.empty:
        raise ValueError('Training and validation splits must both contain images')
    if use_noise_augmentation:
        train_dataset = NoiseAugmentedDataset(image_path, train_data, size, weight, config)
    else:
        train_dataset = DiabeticRetinopathyDataset(image_path, train_data, size, weight, True, config)
    valid_dataset = DiabeticRetinopathyDataset(image_path, valid_data, size, 1.0, False, config)
    print(f'Train Images: {len(train_dataset)}, Valid Images: {len(valid_dataset)}')
    return train_dataset, valid_dataset
