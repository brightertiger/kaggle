import torch
import random
import math
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.augmentations import functional as F
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensor
from torchvision import transforms

def resized_crop(image, height, width, x_min, y_min, x_max, y_max):
    image = F.crop(image, x_min, y_min, x_max, y_max)
    image = cv2.resize(image, (width, height))
    return image

class RandomResizedCrop(ImageOnlyTransform):

    def __init__(self, height, width, scale=(0.08, 1.0), ratio=(3/4, 4/3), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.scale = scale
        self.ratio = ratio

    def apply(self, image, **params):

        height, width = image.shape[:2]
        area = height * width
        
        for attempt in range(15):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5 and min(self.ratio) <= (h / w) <= max(self.ratio):
                w, h = h, w

            if w <= width and h <= height:
                x_min = random.randint(0, width - w)
                y_min = random.randint(0, height - h)
                return resized_crop(image, self.height, self.width, x_min, y_min, x_min+w, y_min+h)

        min_side = min(height, width)
        x_min = random.randint(0, width - min_side)
        y_min = random.randint(0, height - min_side)
        return resized_crop(image, self.height, self.width, x_min, y_min, x_min+min_side, y_min+min_side)

class TrainDataset(Dataset):

    def __init__(self, path, data):
        self.path = path
        data = data.reset_index(drop=True)
        self.image = data['image'].tolist()
        self.label = np.array(data.iloc[:,-6:])
        transform = []
        transform.append(RandomResizedCrop(height=512, width=512, scale=(0.7,1.0), p=1.0))
        transform.append(A.HorizontalFlip())
        transform.append(ToTensor())
        self.transform = A.Compose(transform)
        return None
    
    def __len__(self):
        return len(self.image)

    def __image__(self, image_idx):  
        name = self.path + '/ID_' + image_idx + '.npz'
        image = np.load(name)['arr_0'].astype(float)
        augment = self.transform(image=image)
        image = augment['image']
        return image

    def __label__(self, label_idx):
        label = self.label[label_idx,:]
        label = torch.from_numpy(label).reshape(-1,6)
        return label
    
    def __getitem__(self, idx):
        image = self.__image__(self.image[idx])
        label = self.__label__(idx)
        sample = {'idx': self.image[idx], 'image': image, 'label': label}
        return sample

class ValidDataset(Dataset):

    def __init__(self, path, data):
        self.path = path
        data = data.reset_index(drop=True)
        self.image = data['image'].tolist()
        self.label = np.array(data.iloc[:,-6:])
        transform = []
        transform.append(RandomResizedCrop(height=512, width=512, scale=(0.7,1.0), p=1.0))
        transform.append(ToTensor())
        self.transform = A.Compose(transform)
        return None
    
    def __len__(self):
        return len(self.image)

    def __image__(self, image_idx):  
        name = self.path + '/ID_' + image_idx + '.npz'
        image = np.load(name)['arr_0'].astype(float)
        augment = self.transform(image=image)
        image = augment['image']
        return image

    def __label__(self, label_idx):
        label = self.label[label_idx,:]
        label = torch.from_numpy(label).reshape(-1,6)
        return label
    
    def __getitem__(self, idx):
        image = self.__image__(self.image[idx])
        label = self.__label__(idx)
        sample = {'idx': self.image[idx], 'image': image, 'label': label}
        return sample    
    
def trainLoader(image_path, label_path, fold_idx):
    data = pd.read_csv(label_path)
    train = data[data['fold'] != fold_idx].reset_index(drop=True)
    valid = data[data['fold'] == fold_idx].reset_index(drop=True)
    train = TrainDataset(image_path, train)
    valid = ValidDataset(image_path, valid)
    print('Train Images:', len(train), 'Valid Images:', len(valid))
    return train, valid

def scoreLoader(image_path, label_path):
    data = pd.read_csv(label_path)
    score = ValidDataset(image_path, data)
    print('Score Images:', len(score))
    return score