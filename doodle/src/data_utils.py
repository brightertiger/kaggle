import torch
import cv2
import numpy as np
import pandas as pd
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional


class DoodleDataset(Dataset):
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 category_mapping: List[str], 
                 image_size: int = 64,
                 is_training: bool = True,
                 horizontal_flip_prob: float = 0.5):
        self.drawings = dataframe['drawing'].tolist()
        self.key_ids = dataframe['key_id'].tolist()
        self.category_mapping = category_mapping
        self.image_size = image_size
        self.is_training = is_training
        self.horizontal_flip_prob = horizontal_flip_prob
        
        if 'word' in dataframe.columns:
            self.labels = dataframe['word'].tolist()
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.key_ids)

    def _drawing_to_image(self, drawing_data: str) -> np.ndarray:
        drawing = literal_eval(drawing_data)
        image = np.zeros((256, 256), dtype=np.uint8)
        
        for stroke_idx, stroke in enumerate(drawing):
            stroke_color = 255 - min(stroke_idx, 10) * 13
            
            for point_idx in range(len(stroke[0]) - 1):
                x1, y1 = stroke[0][point_idx], stroke[1][point_idx]
                x2, y2 = stroke[0][point_idx + 1], stroke[1][point_idx + 1]
                cv2.line(image, (x1, y1), (x2, y2), stroke_color, 6)
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        if self.is_training and np.random.uniform() > self.horizontal_flip_prob:
            image = np.fliplr(image)
        
        image = np.atleast_3d(image)
        image = (image / 255.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        
        return image

    def __getitem__(self, idx: int) -> dict:
        image = self._drawing_to_image(self.drawings[idx])
        image = torch.from_numpy(image)
        
        sample = {'image': image, 'key_id': self.key_ids[idx]}
        
        if self.labels is not None:
            label_idx = self.category_mapping.index(self.labels[idx])
            sample['label'] = torch.tensor(label_idx, dtype=torch.long)
        
        return sample


def create_dataloaders(train_df: pd.DataFrame,
                      valid_df: pd.DataFrame,
                      category_mapping: List[str],
                      config) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = DoodleDataset(
        train_df, 
        category_mapping, 
        config.image_size, 
        is_training=True
    )
    
    valid_dataset = DoodleDataset(
        valid_df, 
        category_mapping, 
        config.image_size, 
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader


def create_test_dataloader(test_df: pd.DataFrame,
                          category_mapping: List[str],
                          config) -> DataLoader:
    
    test_dataset = DoodleDataset(
        test_df,
        category_mapping,
        config.image_size,
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return test_loader
