import cv2
import numpy as np
import pandas as pd
import albumentations as A
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from .config import Config

class MelanomaDataset(Dataset):
    def __init__(self, image_path, metadata_df, fold=None, is_training=True):
        self.image_path = image_path
        self.metadata_df = metadata_df
        self.fold = fold
        self.is_training = is_training
        self.transform = self._get_transforms()
        
        if fold is not None:
            if is_training:
                self.data = metadata_df[metadata_df['fold'] != fold].reset_index(drop=True)
            else:
                self.data = metadata_df[metadata_df['fold'] == fold].reset_index(drop=True)
        else:
            self.data = metadata_df.reset_index(drop=True)
    
    def _get_transforms(self):
        if self.is_training:
            transforms = [
                A.RandomSizedCrop(
                    min_max_height=(400, 500), 
                    height=Config.IMAGE_SIZE, 
                    width=Config.IMAGE_SIZE, 
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Cutout(
                    num_holes=Config.CUTOUT_HOLES, 
                    max_h_size=Config.CUTOUT_SIZE, 
                    max_w_size=Config.CUTOUT_SIZE, 
                    fill_value=0, 
                    p=0.5
                ),
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE, p=1.0),
                A.Normalize(),
                ToTensorV2()
            ]
        else:
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE, p=1.0),
                A.Normalize(),
                ToTensorV2()
            ]
        
        return A.Compose(transforms, p=1.0)
    
    def _apply_hair_mask(self, image):
        if np.random.random() <= 0.1:
            hair_transforms = A.Compose([
                A.ShiftScaleRotate(
                    rotate_limit=[-45, 45],
                    scale_limit=[-0.1, 0.1],
                    shift_limit=[-0.1, 0.15],
                    border_mode=3,
                    value=0,
                    p=1.0
                )
            ])
            
            hair_mask = np.random.choice(np.arange(7), 1, p=[0.2, 0.2, 0.22, 0.15, 0.14, 0.06, 0.03])[0]
            mask = np.ones((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            mask = hair_transforms(image=mask)['image']
            mask = cv2.resize(mask/255, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), cv2.INTER_CUBIC)
            mask[mask == 1.0] = 255
            mask[mask != 255.0] = 0
            mask = mask.astype(np.uint8)
            
            image = cv2.bitwise_and(image, image, mask=mask)
        
        return image
    
    def _encode_metadata(self, row):
        age = Config.AGE_LOOKUP.get(row['age_approx'], 0)
        sex = Config.SEX_LOOKUP.get(row['sex'], 0)
        anatomy = Config.ANATOMY_LOOKUP.get(row['anatom_site_general_challenge'], 0)
        
        age_arr = np.zeros(7)
        sex_arr = np.zeros(2)
        anatomy_arr = np.zeros(4)
        
        if age:
            age_arr[age-1] = 1
        if sex:
            sex_arr[sex-1] = 1
        if anatomy:
            anatomy_arr[anatomy-1] = 1
        
        return np.concatenate([age_arr, sex_arr, anatomy_arr])
    
    def _encode_label(self, diagnosis):
        label = np.zeros(Config.NUM_CLASSES)
        label[Config.DIAGNOSIS_LOOKUP[diagnosis]] = 1.0
        return label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        image_path = self.image_path / f"{row['image_id']}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_training:
            image = self._apply_hair_mask(image)
        
        image = self.transform(image=image)['image']
        
        metadata = self._encode_metadata(row)
        
        if 'diagnosis' in row:
            label = self._encode_label(row['diagnosis'])
            return {
                'image': image,
                'metadata': metadata,
                'label': label,
                'image_id': row['image_id']
            }
        else:
            return {
                'image': image,
                'metadata': metadata,
                'image_id': row['image_id']
            }

def create_data_loaders(image_path, metadata_df, fold, batch_size=Config.BATCH_SIZE, num_workers=3):
    train_dataset = MelanomaDataset(image_path, metadata_df, fold, is_training=True)
    valid_dataset = MelanomaDataset(image_path, metadata_df, fold, is_training=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, valid_loader

def load_metadata(data_path):
    train_metadata = pd.read_csv(data_path / 'train_metadata.csv')
    test_metadata = pd.read_csv(data_path / 'test_metadata.csv')
    
    train_metadata['sex'] = train_metadata['sex'].fillna('unknown')
    train_metadata['age_approx'] = train_metadata['age_approx'].fillna(0).round(-1).clip(20, 80)
    train_metadata['anatom_site_general_challenge'] = train_metadata['anatom_site_general_challenge'].fillna('unknown')
    
    test_metadata['sex'] = test_metadata['sex'].fillna('unknown')
    test_metadata['age_approx'] = test_metadata['age_approx'].fillna(0).round(-1).clip(20, 80)
    test_metadata['anatom_site_general_challenge'] = test_metadata['anatom_site_general_challenge'].fillna('unknown')
    
    return train_metadata, test_metadata
