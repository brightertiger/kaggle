import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from typing import Tuple, List
import zipfile
import warnings

from .config import Config


class IMetDataset(Dataset):
    def __init__(self, 
                 image_path: str, 
                 data: pd.DataFrame, 
                 config: Config, 
                 is_training: bool = True):
        self.image_path = image_path
        self.data = data.reset_index(drop=True)
        self.config = config
        self.is_training = is_training
        
        self.image_ids = self.data['id'].tolist()
        self.labels = self.data['attribute_ids'].fillna('').astype(str).tolist()
        
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        settings = self.config.train_transforms if self.is_training else self.config.val_transforms
        if self.is_training:
            transform_list = [
                transforms.RandomHorizontalFlip(p=self.config.train_transforms['random_horizontal_flip']),
                (transforms.RandomCrop(self.config.image_size, pad_if_needed=True)
                 if settings.get('random_crop', True) else transforms.CenterCrop(self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.train_transforms['normalize']['mean'],
                    std=self.config.train_transforms['normalize']['std']
                )
            ]
        else:
            transform_list = [
                (transforms.RandomCrop(self.config.image_size, pad_if_needed=True)
                 if settings.get('random_crop', True) else transforms.CenterCrop(self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.val_transforms['normalize']['mean'],
                    std=self.config.val_transforms['normalize']['std']
                )
            ]
        
        return transforms.Compose(transform_list)
    
    def _load_image(self, image_id: str) -> torch.Tensor:
        image_path = os.path.join(self.image_path, f'{image_id}.png')
        try:
            with Image.open(image_path) as source:
                image = source.convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            raise RuntimeError(f'Cannot load image: {image_path}') from e
    
    def _encode_labels(self, label_str: str) -> torch.Tensor:
        label_ids = label_str.split()
        label_array = np.full(self.config.num_classes, self.config.epsilon / 1000, dtype=np.float32)
        
        for label_id in label_ids:
            idx = int(label_id)
            if not 0 <= idx < self.config.num_classes:
                raise ValueError(f'Attribute {idx} is outside the configured class range')
            label_array[idx] = 1 - self.config.epsilon

        return torch.from_numpy(label_array)

    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        image = self._load_image(image_id)
        label = self._encode_labels(self.labels[idx])
        
        return {
            'idx': image_id,
            'image': image,
            'label': label
        }


class IMetTestDataset(Dataset):
    def __init__(self, image_path: str, config: Config):
        self.image_path = image_path
        self.config = config
        
        sample_submission = pd.read_csv(config.sample_submission_path, dtype={'id': str})
        self.image_ids = sample_submission['id'].tolist()
        
        self.transform = transforms.Compose([
            (transforms.RandomCrop(config.image_size, pad_if_needed=True)
             if config.val_transforms.get('random_crop', True) else transforms.CenterCrop(config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.val_transforms['normalize']['mean'],
                std=config.val_transforms['normalize']['std']
            )
        ])
    
    def _load_image(self, image_id: str) -> torch.Tensor:
        image_path = os.path.join(self.image_path, f'{image_id}.png')
        try:
            with Image.open(image_path) as source:
                image = source.convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            raise RuntimeError(f'Cannot load image: {image_path}') from e
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        image = self._load_image(image_id)
        
        return {
            'idx': image_id,
            'image': image
        }


class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config
    
    def load_train_data(self) -> pd.DataFrame:
        path = self.config.train_csv_path
        if not os.path.exists(path):
            raise FileNotFoundError(f'Training data not found at {path}')
        kwargs = {'dtype': {'id': str, 'attribute_ids': str}, 'keep_default_na': False}
        if path.endswith('.zip'):
            with zipfile.ZipFile(path) as archive:
                files = [name for name in archive.namelist() if name.endswith('.csv') and not name.startswith('__MACOSX/')]
                if len(files) != 1:
                    raise ValueError('train.csv.zip must contain exactly one CSV')
                with archive.open(files[0]) as source:
                    data = pd.read_csv(source, **kwargs)
        else:
            data = pd.read_csv(path, **kwargs)
        if not {'id', 'attribute_ids'}.issubset(data.columns):
            raise ValueError('train.csv requires id and attribute_ids columns')
        if data['id'].duplicated().any():
            raise ValueError('Training image ids must be unique')
        return data

    def load_subset_data(self) -> pd.DataFrame:
        if not os.path.exists(self.config.subset_csv_path):
            warnings.warn('Optional subset.csv is absent; retaining all labels. '
                          'The original long-tail filter requires intent, percent, total.', stacklevel=2)
            return pd.DataFrame(columns=['intent', 'percent', 'total'])
        data = pd.read_csv(self.config.subset_csv_path, dtype={'intent': str})
        if not {'intent', 'percent', 'total'}.issubset(data.columns):
            raise ValueError('subset.csv requires intent, percent and total columns')
        return data

    def clean_labels(self, labels: str, long_tail_classes: List[str]) -> str:
        label_list = labels.split()
        cleaned_labels = [x for x in label_list if x not in long_tail_classes]
        return ' '.join(cleaned_labels)
    
    def get_rare_class(self, labels: str, attribute_counts: Counter) -> str:
        label_list = labels.split()
        if not label_list:
            return '0'
        
        counts = [attribute_counts.get(x, 0) for x in label_list]
        min_count_idx = np.argmin(counts)
        return label_list[min_count_idx]
    
    def create_folds(self) -> pd.DataFrame:
        print("🔄 Loading and preprocessing data...")
        
        data = self.load_train_data()
        print(f"📊 Loaded {len(data)} training samples")
        
        subset_data = self.load_subset_data()
        
        attributes = ' '.join(data['attribute_ids'].tolist()).split()
        attribute_counts = Counter(attributes)
        
        tail_1 = subset_data[(subset_data['percent'] <= 0.2) & (subset_data['total'] <= 200)]
        tail_2 = subset_data[subset_data['total'] <= 20]
        long_tail_classes = pd.concat([tail_1, tail_2], ignore_index=True)['intent'].unique().tolist()
        long_tail_classes = [str(x) for x in long_tail_classes]
        
        print(f"🧹 Cleaning labels (removing {len(long_tail_classes)} long-tail classes)...")
        data['attribute_ids'] = data['attribute_ids'].map(
            lambda x: self.clean_labels(x, long_tail_classes)
        )
        
        data = data[data['attribute_ids'].str.len() > 0].reset_index(drop=True)
        
        data['class'] = data['attribute_ids'].map(
            lambda x: self.get_rare_class(x, attribute_counts)
        )
        
        print(f"📊 After cleaning: {len(data)} samples")
        print(f"📊 Unique attributes: {len(attribute_counts)}")
        
        print("🔄 Creating stratified folds...")
        folds = StratifiedKFold(
            n_splits=self.config.num_folds, 
            shuffle=False
        )
        
        data['fold'] = 0
        for fold_idx, (_, idx) in enumerate(folds.split(data.index, data['class'])):
            data.loc[idx, 'fold'] = fold_idx + 1
        
        fold_counts = data['fold'].value_counts().sort_index()
        print(f"📊 Fold distribution: {dict(fold_counts)}")
        
        data.to_csv(self.config.folds_csv_path, index=False)
        print(f"💾 Saved folds to {self.config.folds_csv_path}")
        
        return data


def create_data_loaders(config: Config, fold_idx: int) -> Tuple[DataLoader, DataLoader]:
    if not os.path.exists(config.folds_csv_path):
        raise FileNotFoundError(f"Folds file not found. Run preprocessing first: {config.folds_csv_path}")
    
    data = pd.read_csv(config.folds_csv_path, dtype={'id': str, 'attribute_ids': str}, keep_default_na=False)
    if set(data['fold'].unique()) != set(range(1, config.num_folds + 1)):
        raise ValueError('folds.csv does not match num_folds; rerun preprocessing')
    
    train_data = data[data['fold'] != fold_idx].reset_index(drop=True)
    valid_data = data[data['fold'] == fold_idx].reset_index(drop=True)
    
    if train_data.empty or valid_data.empty:
        raise ValueError(f'Fold {fold_idx} requires nonempty train and validation sets')

    train_dataset = IMetDataset(config.train_images_path, train_data, config, is_training=True)
    valid_dataset = IMetDataset(config.train_images_path, valid_data, config, is_training=False)
    
    print(f"📊 Train: {len(train_dataset)} samples, Valid: {len(valid_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=config.device.startswith('cuda')
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=max(config.batch_size // 2, 1),
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=config.device.startswith('cuda')
    )
    
    return train_loader, valid_loader


def create_test_loader(config: Config) -> DataLoader:
    test_dataset = IMetTestDataset(config.test_images_path, config)
    
    print(f"📊 Test: {len(test_dataset)} samples")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=config.device.startswith('cuda')
    )
    
    return test_loader
