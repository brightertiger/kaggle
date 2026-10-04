"""Competition image schema, shared label vocabulary, and identity-aware splits."""
import os
from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class WhaleDataset(Dataset):
    def __init__(self, image_path: str, data: pd.DataFrame, size: int,
                 transform: bool = True, is_test: bool = False, seed: int = 42):
        self.image_path = image_path
        self.size = (size, size)
        self.is_test = is_test
        self.transform = self._get_train_transforms() if transform else self._get_valid_transforms()
        self.transform.set_random_seed(seed)
        self.images = data['Image'].tolist()
        self.labels = data['Id'].tolist() if not is_test else None

    def _get_train_transforms(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1),
                     rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_valid_transforms(self):
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def _load_image(self, image_name):
        with Image.open(os.path.join(self.image_path, image_name)) as image:
            image = image.convert('RGB').resize(self.size, resample=Image.Resampling.BICUBIC)
            array = self.transform(image=np.array(image))['image']
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()

    def __getitem__(self, idx):
        item = {'image': self._load_image(self.images[idx]), 'image_name': self.images[idx]}
        if not self.is_test:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def read_labels(csv_path):
    data = pd.read_csv(csv_path, dtype={'Image': str, 'Id': str})
    if not {'Image', 'Id'}.issubset(data.columns) or data[['Image', 'Id']].isna().any().any():
        raise ValueError(f"{csv_path} must contain nonempty Image and Id columns")
    if data['Image'].duplicated().any():
        raise ValueError(f"Duplicate Image entries in {csv_path}")
    return data[data.Id != 'new_whale'].reset_index(drop=True)


def encode_labels(data, class_names):
    mapping = {name: idx for idx, name in enumerate(class_names)}
    encoded = data.copy()
    encoded['Id'] = data.Id.map(mapping)
    if encoded.Id.isna().any():
        unknown = sorted(set(data.loc[encoded.Id.isna(), 'Id']))
        raise ValueError(f"IDs outside training vocabulary: {unknown}")
    encoded['Id'] = encoded.Id.astype(int)
    return encoded


def split_labels(train_csv_path, class_names=None, val_csv_path=None, return_valid=True):
    data = read_labels(train_csv_path)
    class_names = sorted(data.Id.unique().tolist()) if class_names is None else list(class_names)
    if not class_names:
        raise ValueError("No known whale IDs in training data")
    data = encode_labels(data, class_names)
    if not return_valid:
        return data, None, class_names
    if val_csv_path:
        valid = encode_labels(read_labels(val_csv_path), class_names)
        if set(data.Image) & set(valid.Image):
            raise ValueError("Training and validation images overlap")
        return data, valid if len(valid) else None, class_names
    # Preserve the original split: first image of each repeated identity held out.
    counts = data.Id.value_counts()
    valid_images = set(data[data.Id.map(counts) > 1].groupby('Id').first().Image)
    train = data[~data.Image.isin(valid_images)].reset_index(drop=True)
    valid = data[data.Image.isin(valid_images)].reset_index(drop=True)
    return train, valid if len(valid) else None, class_names


def _loader(config, dataset, training=False, sampler=None):
    if training and len(dataset) < 2:
        raise ValueError("At least two training examples are required for BatchNorm")
    if not len(dataset):
        raise ValueError("No images found")
    # Avoid singleton training batches, which BatchNorm1d cannot normalize.
    return DataLoader(dataset, batch_size=config.batch_size,
                      shuffle=training and sampler is None, sampler=sampler,
                      drop_last=training and len(dataset) % config.batch_size == 1,
                      num_workers=config.num_workers, pin_memory=config.device.startswith('cuda'),
                      generator=torch.Generator().manual_seed(config.seed))


def create_data_loaders(config, train_csv_path: str, image_dir: str,
                        return_valid: bool = True, class_names=None,
                        val_csv_path=None) -> Tuple[DataLoader, Optional[DataLoader]]:
    train, valid, names = split_labels(train_csv_path, class_names, val_csv_path, return_valid)
    dataset = WhaleDataset(image_dir, train, config.image_size,
                           transform=config.use_augmentation, seed=config.seed)
    dataset.class_names = names
    sampler = None
    if config.use_weighted_sampling:
        weights = 1 / train.Id.map(train.Id.value_counts()).to_numpy()
        sampler = WeightedRandomSampler(torch.as_tensor(weights), len(train), replacement=True,
                                        generator=torch.Generator().manual_seed(config.seed))
    train_loader = _loader(config, dataset, training=True, sampler=sampler)
    valid_loader = None
    if valid is not None:
        valid_loader = _loader(config, WhaleDataset(image_dir, valid, config.image_size, transform=False))
    print(f'Train Images: {len(train)}, Valid Images: {0 if valid is None else len(valid)}')
    return train_loader, valid_loader


def create_test_loader(config, test_image_dir: str, test_csv_path=None):
    if test_csv_path:
        data = pd.read_csv(test_csv_path, dtype={'Image': str})
        if 'Image' not in data or data.Image.isna().any() or data.Image.duplicated().any():
            raise ValueError("Test CSV requires unique, nonempty Image entries")
    else:
        files = sorted(p.name for p in Path(test_image_dir).iterdir()
                       if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        data = pd.DataFrame({'Image': files})
    return _loader(config, WhaleDataset(test_image_dir, data, config.image_size,
                                       transform=False, is_test=True))


def create_pseudo_label_loader(config, pseudo_csv_path, image_dir, class_names=None,
                               excluded_images=()):
    data = read_labels(pseudo_csv_path)
    data = data[~data.Image.isin(excluded_images)]
    if 'confidence' in data:
        data = data[data.confidence >= config.pseudo_label_threshold]
    if class_names is None:
        class_names = sorted(read_labels(config.train_csv).Id.unique().tolist())
    data = encode_labels(data, class_names)
    return _loader(config, WhaleDataset(image_dir, data, config.image_size,
                                       transform=config.use_augmentation, seed=config.seed), training=True)


class WhalePairDataset(Dataset):
    """Balanced positive/negative pairs; validation anchors are held-out images."""
    def __init__(self, anchors, references, seed=42):
        self.anchors, self.references = anchors, references
        rng = np.random.default_rng(seed)
        groups = {}
        for j, label in enumerate(references.labels):
            groups.setdefault(label, []).append(j)
        labels = sorted(groups)
        if len(labels) < 2:
            raise ValueError("Siamese training requires at least two known identities")
        label_positions = {label: i for i, label in enumerate(labels)}
        self.pairs = []
        for i, label in enumerate(anchors.labels):
            positive = [j for j in groups[label] if references.images[j] != anchors.images[i]]
            if not positive:
                continue
            self.pairs.append((i, int(rng.choice(positive)), 1.0))
            negative_position = int(rng.integers(len(labels) - 1))
            negative_position += negative_position >= label_positions[label]
            negative_label = labels[negative_position]
            self.pairs.append((i, int(rng.choice(groups[negative_label])), 0.0))
        if not self.pairs:
            raise ValueError("Siamese pairs require repeated identities after the validation split")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]
        return {'image1': self.anchors[i]['image'], 'image2': self.references[j]['image'],
                'label': torch.tensor([label], dtype=torch.float32)}


def create_pair_loaders(config, train_csv_path, image_dir, class_names, val_csv_path=None):
    train, valid, _ = split_labels(train_csv_path, class_names, val_csv_path)
    augmented = WhaleDataset(image_dir, train, config.image_size,
                             transform=config.use_augmentation, seed=config.seed)
    train_loader = _loader(config, WhalePairDataset(augmented, augmented, config.seed), training=True)
    valid_loader = None
    if valid is not None:
        anchors = WhaleDataset(image_dir, valid, config.image_size, transform=False)
        references = WhaleDataset(image_dir, train, config.image_size, transform=False)
        valid_loader = _loader(config, WhalePairDataset(anchors, references, config.seed))
    return train_loader, valid_loader
