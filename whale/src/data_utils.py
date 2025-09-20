import torch
import albumentations as A
import numpy as np
import pandas as pd
import glob
import PIL
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from typing import Tuple, Optional, List
import os

class WhaleDataset(Dataset):
    def __init__(self, image_path: str, data: pd.DataFrame, size: int, 
                 transform: bool = True, is_test: bool = False):
        self.image_path = image_path
        self.size = (size, size)
        self.is_test = is_test
        
        if transform:
            self.transform = self._get_train_transforms()
        else:
            self.transform = self._get_valid_transforms()
            
        data = data.reset_index(drop=True)
        self.images = data['Image'].tolist()
        self.labels = data['Id'].tolist() if not is_test else None
        
    def _get_train_transforms(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_valid_transforms(self):
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def _load_image(self, image_name: str) -> np.ndarray:
        image_path = os.path.join(self.image_path, image_name)
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.size, resample=PIL.Image.BICUBIC)
        image = self.transform(image=np.array(image))
        image = image['image']
        image = image.transpose(2, 0, 1)
        return image
    
    def __getitem__(self, idx):
        image = self._load_image(self.images[idx])
        image = torch.from_numpy(image).float()
        
        if self.is_test:
            return {'image': image, 'image_name': self.images[idx]}
        
        label = self.labels[idx]
        label = np.array(label).reshape(1, 1, 1)
        label = torch.from_numpy(label).long()
        
        return {'image': image, 'label': label, 'image_name': self.images[idx]}

def create_data_loaders(config, train_csv_path: str, image_dir: str, 
                       return_valid: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
    encoder = LabelEncoder()
    data = pd.read_csv(train_csv_path)
    
    # Remove 'new_whale' class for training
    data = data[data['Id'] != 'new_whale'].reset_index(drop=True)
    data['Id'] = encoder.fit_transform(data['Id'])
    
    # Create validation split based on whale ID (stratified)
    count = data.Id.value_counts()
    count.name = 'count'
    data = data.join(count, on='Id')
    
    # Use whales with >1 image for validation
    valid_images = set(data[(data['count'] > 1)].groupby('Id').first().Image)
    train_data = data[~data['Image'].isin(valid_images)].reset_index(drop=True)
    valid_data = data[data['Image'].isin(valid_images)].reset_index(drop=True)
    
    if not return_valid:
        train_data = pd.concat([train_data, valid_data]).reset_index(drop=True)
        valid_data = None
    
    # Create datasets
    train_dataset = WhaleDataset(image_dir, train_data, config.image_size, 
                                transform=True, is_test=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    if valid_data is not None:
        valid_dataset = WhaleDataset(image_dir, valid_data, config.image_size, 
                                    transform=False, is_test=False)
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        print(f'Train Images: {len(train_dataset)}, Valid Images: {len(valid_dataset)}')
        return train_loader, valid_loader
    else:
        print(f'Train Images: {len(train_dataset)}')
        return train_loader, None

def create_test_loader(config, test_image_dir: str) -> DataLoader:
    files = glob.glob(os.path.join(test_image_dir, '*.jpg'))
    files = [os.path.basename(f) for f in files]
    
    data = pd.DataFrame({'Image': files, 'Id': [0] * len(files)})
    print(f'Test Images: {len(data)}')
    
    dataset = WhaleDataset(test_image_dir, data, config.image_size, 
                          transform=False, is_test=True)
    
    return DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

def create_pseudo_label_loader(config, pseudo_csv_path: str, 
                              image_dir: str) -> DataLoader:
    data = pd.read_csv(pseudo_csv_path)
    data['count'] = 1 / (2 + data['count'])
    
    dataset = WhaleDataset(image_dir, data, config.image_size, 
                          transform=True, is_test=False)
    
    return DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
