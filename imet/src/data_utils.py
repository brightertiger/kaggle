import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from typing import Tuple, List, Optional
import zipfile

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
        self.labels = self.data['attribute_ids'].tolist()
        
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        if self.is_training:
            transform_list = [
                transforms.RandomHorizontalFlip(p=self.config.train_transforms['random_horizontal_flip']),
                transforms.RandomCrop(size=(self.config.image_size, self.config.image_size), pad_if_needed=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.train_transforms['normalize']['mean'],
                    std=self.config.train_transforms['normalize']['std']
                )
            ]
        else:
            transform_list = [
                transforms.RandomCrop(size=(self.config.image_size, self.config.image_size), pad_if_needed=True),
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
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            return torch.zeros(3, self.config.image_size, self.config.image_size)
    
    def _encode_labels(self, label_str: str) -> torch.Tensor:
        label_ids = label_str.split()
        label_array = np.full(self.config.num_classes, self.config.epsilon / 1000, dtype=np.float32)
        
        for label_id in label_ids:
            try:
                idx = int(label_id)
                if 0 <= idx < self.config.num_classes:
                    label_array[idx] = 1 - self.config.epsilon
            except ValueError:
                continue
        
        return torch.from_numpy(label_array).reshape(1, -1)
    
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
        
        sample_submission = pd.read_csv(config.sample_submission_path)
        self.image_ids = sample_submission['id'].tolist()
        
        self.transform = transforms.Compose([
            transforms.RandomCrop(size=(config.image_size, config.image_size), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.val_transforms['normalize']['mean'],
                std=config.val_transforms['normalize']['std']
            )
        ])
    
    def _load_image(self, image_id: str) -> torch.Tensor:
        image_path = os.path.join(self.image_path, f'{image_id}.png')
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            return torch.zeros(3, self.config.image_size, self.config.image_size)
    
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
        if os.path.exists(self.config.train_csv_path):
            if self.config.train_csv_path.endswith('.zip'):
                with zipfile.ZipFile(self.config.train_csv_path, 'r') as zip_ref:
                    csv_file = zip_ref.namelist()[0]
                    return pd.read_csv(zip_ref.open(csv_file))
            else:
                return pd.read_csv(self.config.train_csv_path)
        else:
            raise FileNotFoundError(f"Training data not found at {self.config.train_csv_path}")
    
    def load_subset_data(self) -> pd.DataFrame:
        if os.path.exists(self.config.subset_csv_path):
            return pd.read_csv(self.config.subset_csv_path)
        else:
            raise FileNotFoundError(f"Subset data not found at {self.config.subset_csv_path}")
    
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
        long_tail_classes = tail_1.append(tail_2)['intent'].unique().tolist()
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
            shuffle=False, 
            random_state=self.config.seed
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
    
    data = pd.read_csv(config.folds_csv_path)
    
    train_data = data[data['fold'] != fold_idx].reset_index(drop=True)
    valid_data = data[data['fold'] == fold_idx].reset_index(drop=True)
    
    train_dataset = IMetDataset(config.train_images_path, train_data, config, is_training=True)
    valid_dataset = IMetDataset(config.train_images_path, valid_data, config, is_training=False)
    
    print(f"📊 Train: {len(train_dataset)} samples, Valid: {len(valid_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=max(config.batch_size // 2, 1),
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True
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
        pin_memory=True
    )
    
    return test_loader
