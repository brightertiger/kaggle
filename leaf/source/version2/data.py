import torch
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2

def trainTF():
    transforms = []
    transforms += [A.Transpose(p=0.5)]
    transforms += [A.HorizontalFlip(p=0.5)]
    transforms += [A.VerticalFlip(p=0.5)]
    transforms += [A.RandomRotate90(p=0.5)]
    transforms += [A.ShiftScaleRotate(p=0.5)]
    transforms += [A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1),                      p=0.5)]
    transforms += [A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2,                      p=0.5)]
    transforms += [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],                                        max_pixel_value=255.0)]
    transforms += [A.CoarseDropout(p=0.5)]
    transforms += [A.Cutout(p=0.5)]
    transforms += [A.Resize(500,500, p=1.)]
    transforms += [A.RandomResizedCrop(384, 384, p=1.0)]
    transforms += [ToTensorV2()]
    transforms = A.Compose(transforms, p=1.)
    return transforms

def validTF():
    transforms = []
    transforms += [A.Resize(500,500, p=1.)]
    transforms += [A.CenterCrop(384, 384, p=1.)]
    transforms += [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],                                        max_pixel_value=255.0)]
    transforms += [ToTensorV2()]
    transforms = A.Compose(transforms, p=1.)
    return transforms

class TrainDataset(Dataset):

    def __init__(self, path, data, fold):
        self.path = path
        self.data = data[(data['fold'] != fold)]
        self.data = self.data.reset_index(drop=True)
        self.transform = trainTF()
        return None
    
    def labels(self):
        return list(self.data.label.values)
    
    def __len__(self):
        return len(self.data)

    def __image__(self, idx):
        image = self.data.loc[idx, 'image_id']
        name = self.path + '/' + image
        image = cv2.imread(name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        return image

    def __label__(self, idx):
        label = self.data.loc[idx,'label']
        array = torch.tensor([label]).reshape(1,1).long()
        return array
    
    def __getitem__(self, idx):
        image = self.__image__(idx)
        label = self.__label__(idx)
        sample = (image, label)
        return sample

class ValidDataset(Dataset):

    def __init__(self, path, data, fold):
        self.path = path
        self.transform = validTF()
        self.data = data[(data['fold'] == fold)]
        self.data = self.data.reset_index(drop=True)
        return None
    
    def __len__(self):
        return len(self.data)

    def __image__(self, idx):
        image = self.data.loc[idx, 'image_id']
        name = self.path + '/' + image
        image = cv2.imread(name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        return image

    def __label__(self, idx):
        label = self.data.loc[idx,'label']
        array = torch.tensor([label]).reshape(1,1).long()
        return array
    
    def __getitem__(self, idx):
        image = self.__image__(idx)
        label = self.__label__(idx)
        sample = (image, label)
        return sample  
    
def trainLoader(image_path, label_path, fold_idx):
    data = pd.read_csv(label_path)
    train = TrainDataset(image_path, data, fold_idx)
    valid = ValidDataset(image_path, data, fold_idx)
    print('Train Images:', len(train), 'Valid Images:', len(valid))
    return train, valid