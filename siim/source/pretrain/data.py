import torch
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2

def trainTF():
    transforms = []
    transforms += [A.RandomSizedCrop(min_max_height=(400, 500), height=512, width=512, p=0.5)]
    transforms += [A.HorizontalFlip(p=0.5)]
    transforms += [A.VerticalFlip(p=0.5)]
    transforms += [A.RandomRotate90(p=0.5)]
    transforms += [A.Resize(height=512, width=512, p=1.)]
    transforms += [A.Cutout(num_holes=16, max_h_size=64, max_w_size=64, fill_value=0, p=0.5)]
    transforms += [A.Normalize()]
    transforms += [ToTensorV2()]
    transforms = A.Compose(transforms, p=1.)
    return transforms

def validTF():
    transforms = []
    transforms += [A.HorizontalFlip(p=0.5)]
    transforms += [A.VerticalFlip(p=0.5)]
    transforms += [A.Resize(height=512, width=512, p=1.)]
    transforms += [A.Normalize()]
    transforms += [ToTensorV2()]
    transforms = A.Compose(transforms, p=1.)
    return transforms

LOOKUP = {'other' : 0, 'melanoma' : 1, 'nevus' : 2, 'keratosis' : 3}

class TrainDataset(Dataset):

    def __init__(self, path, data, fold):
        self.path = path
        self.data = data[(data['fold'] != fold)]
        self.data = self.data.reset_index(drop=True)
        self.transform = trainTF()
        return None
    
    def labels(self):
        return list(self.data.target.values)
    
    def __len__(self):
        return len(self.data)

    def __image__(self, idx):
        image = self.data.loc[idx, 'image_id']
        name = self.path + '/' + image
        image = cv2.imread(name, cv2.IMREAD_COLOR)
        image = self.transform(image=image)['image']
        return image

    def __label__(self, idx):
        label = self.data.loc[idx,'diagnosis']
        array = [0.,0.,0.,0.]
        array[LOOKUP[label]] += 1.
        array = np.array(array).astype(float).reshape(4,)
        return array
    
    def __getitem__(self, idx):
        image = self.__image__(idx)
        label = self.__label__(idx)
        sample = {'image': image, 'label': label}
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
        image = self.transform(image=image)['image']
        return image

    def __label__(self, idx):
        label = self.data.loc[idx,'diagnosis']
        array = [0.,0.,0.,0.]
        array[LOOKUP[label]] += 1.
        array = np.array(array).astype(float).reshape(4,)
        return array
    
    def __getitem__(self, idx):
        image = self.__image__(idx)
        label = self.__label__(idx)
        sample = {'image': image, 'label': label}
        return sample   
    
def trainLoader(image_path, label_path, fold_idx):
    data = pd.read_csv(label_path)
    train = TrainDataset(image_path, data, fold_idx)
    valid = ValidDataset(image_path, data, fold_idx)
    print('Train Images:', len(train), 'Valid Images:', len(valid))
    return train, valid