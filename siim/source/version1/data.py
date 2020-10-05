import torch
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from collections import defaultdict

SEX_LOOKUP = defaultdict(int,{'male':1, 'female':2})
AGE_LOOKUP = defaultdict(int,{20:1, 30:2, 40:3, 50:4, 60:5, 70:6, 80:7})
ANT_LOOKUP = defaultdict(int,{'lower extremity':1, 'upper extremity':2, 'torso':3, 'head/neck':4})

def trainTF():
    transforms = []
    transforms += [A.RandomSizedCrop(min_max_height=(400, 400), height=512, width=512, p=0.5)]
    transforms += [A.HorizontalFlip(p=0.5)]
    transforms += [A.VerticalFlip(p=0.5)]
    transforms += [A.RandomRotate90(p=0.5)]
    transforms += [A.Resize(height=512, width=512, p=1)]
    transforms += [A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5)]
    transforms += [A.Normalize()]
    transforms += [ToTensorV2()]
    transforms = A.Compose(transforms, p=1.)
    return transforms

def validTF():
    transforms = []
    transforms += [A.HorizontalFlip(p=0.5)]
    transforms += [A.VerticalFlip(p=0.5)]
    transforms += [A.Resize(height=512, width=512, p=1.0)]
    transforms += [A.Normalize()]
    transforms += [ToTensorV2()]
    transforms = A.Compose(transforms, p=1.)
    return transforms

class TrainDataset(Dataset):

    def __init__(self, path, data, fold):
        self.path = path
        self.data = data[data['fold'] != fold].copy()
        self.data['age_approx'] = self.data.age_approx.round(-1).clip(20,80)
        self.data = self.data.reset_index(drop=True)
        self.transform = trainTF()
        return None
    
    def labels(self):
        return list(self.data.target.values)
    
    def __len__(self):
        return len(self.data)

    def __image__(self, idx):
        image = self.data.loc[idx, 'image_id']
        name = self.path + '/' + image + '.jpg'
        image = cv2.imread(name, cv2.IMREAD_COLOR)
        image = self.transform(image=image)['image']
        return image

    def __label__(self, idx):
        label = self.data.loc[idx,'target']
        array = np.array([0.,0.]).astype(np.float64)
        array[label] += 1
        array = torch.from_numpy(array).reshape(2,)
        return array
    
    def __metad__(self, idx):
        age = self.data.loc[idx,'age_approx']
        sex = self.data.loc[idx,'sex']
        ant = self.data.loc[idx,'anatom_site_general_challenge']
        age = AGE_LOOKUP[age]
        sex = SEX_LOOKUP[sex]
        ant = ANT_LOOKUP[ant]
        age_arr = np.array([0] * 7)
        sex_arr = np.array([0] * 2)
        ant_arr = np.array([0] * 4)
        if age: age_arr[age-1] += 1
        if sex: sex_arr[sex-1] += 1
        if ant: ant_arr[ant-1] += 1
        metad = np.hstack([age_arr, sex_arr, ant_arr])
        return metad
        
    def __getitem__(self, idx):
        image = self.__image__(idx)
        label = self.__label__(idx)
        metad = self.__metad__(idx)
        sample = {'image': image, 'label': label, 'metad': metad}
        return sample

class ValidDataset(Dataset):

    def __init__(self, path, data, fold):
        self.path = path
        self.transform = validTF()
        self.data = data[(data['fold'] == fold) & (data['source'] == 'ISIC20')].copy()
        self.data['age_approx'] = self.data.age_approx.round(-1).clip(20,80)
        self.data = self.data.reset_index(drop=True)
        return None
    
    def __len__(self):
        return len(self.data)

    def __image__(self, idx):
        image = self.data.loc[idx, 'image_id']
        name = self.path + '/' + image + '.jpg'
        image = cv2.imread(name, cv2.IMREAD_COLOR)
        image = self.transform(image=image)['image']
        return image

    def __label__(self, idx):
        label = self.data.loc[idx,'target']
        array = np.array([0.,0.]).astype(np.float64)
        array[label] += 1
        array = torch.from_numpy(array).reshape(2,)
        return array
    
    def __metad__(self, idx):
        age = self.data.loc[idx,'age_approx']
        sex = self.data.loc[idx,'sex']
        ant = self.data.loc[idx,'anatom_site_general_challenge']
        age = AGE_LOOKUP[age]
        sex = SEX_LOOKUP[sex]
        ant = ANT_LOOKUP[ant]
        age_arr = np.array([0] * 7)
        sex_arr = np.array([0] * 2)
        ant_arr = np.array([0] * 4)
        if age: age_arr[age-1] += 1
        if sex: sex_arr[sex-1] += 1
        if ant: ant_arr[ant-1] += 1
        metad = np.hstack([age_arr, sex_arr, ant_arr])
        return metad
        
    def __getitem__(self, idx):
        image = self.__image__(idx)
        label = self.__label__(idx)
        metad = self.__metad__(idx)
        sample = {'image': image, 'label': label, 'metad': metad}
        return sample  
    
def trainLoader(image_path, label_path, fold_idx):
    data = pd.read_csv(label_path)
    train = TrainDataset(image_path, data, fold_idx)
    valid = ValidDataset(image_path, data, fold_idx)
    print('Train Images:', len(train), 'Valid Images:', len(valid))
    return train, valid