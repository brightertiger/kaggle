import torch
import albumentations as A
import numpy as np
import pandas as pd
import glob
import PIL
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torch.utils.data import ConcatDataset

TRAIN_TFMS = []
TRAIN_TFMS.append(A.HorizontalFlip(p=0.5))
TRAIN_TFMS.append(A.ShiftScaleRotate(rotate_limit=15, p=0.5))
TRAIN_TFMS.append(A.RandomBrightnessContrast())
TRAIN_TFMS.append(A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
TRAIN_TFMS = A.Compose(TRAIN_TFMS)
VALID_TFMS = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

class PairTrainDataset(Dataset):

    def __init__(self, path, data, size, transform = True):
        self.path = path
        self.size = (size,size)
        if transform : self.transform = TRAIN_TFMS 
        else : self.transform = VALID_TFMS
        self.data = data
        return None

    def __len__(self):
        return self.data.shape[0]

    def __image__(self, image_idx):
        image = Image.open(self.path + '/' + image_idx).convert('RGB')
        image = image.resize(self.size, resample=PIL.Image.BICUBIC)
        image = self.transform(image=np.array(image))
        image = image['image']
        image = image.transpose(2,0,1)
        return image

    def __getitem__(self, idx):
        data = self.data.iloc[idx,:].tolist()
        choice = int(np.random.uniform() > 0.5)
        image_1 = self.__image__(data[choice])
        image_2 = self.__image__(data[1 - choice])
        image_1 = torch.from_numpy(image_1)
        image_2 = torch.from_numpy(image_2)
        label = np.array(data[2]).reshape(1, 1, 1)
        sample = {'image_1':image_1, 'image_2': image_2, 'label': label}
        return sample

class PairScoreDataset(Dataset):

    def __init__(self, train_path, score_path, data, size, transform = True):
        self.train_path = train_path
        self.score_path = score_path
        self.size = (size,size)
        if transform : self.transform = TRAIN_TFMS 
        else : self.transform = VALID_TFMS
        self.data = data
        return None

    def __len__(self):
        return self.data.shape[0]

    def __image__(self, path, image_idx):
        image = Image.open(path + '/' + image_idx).convert('RGB')
        image = image.resize(self.size, resample=PIL.Image.BICUBIC)
        image = self.transform(image=np.array(image))
        image = image['image']
        image = image.transpose(2,0,1)
        return image

    def __getitem__(self, idx):
        data = self.data.iloc[idx,:].tolist()
        idx = [data[0]]
        image_1 = self.__image__(self.score_path, data[0])
        image_2 = self.__image__(self.train_path, data[1])
        image_1 = torch.from_numpy(image_1)
        image_2 = torch.from_numpy(image_2)
        label = np.array(data[2]).reshape(1, 1, 1)
        sample = {'idx':idx, 'image_1':image_1, 'image_2':image_2, 'label':label}
        return sample

def data_loader(train, valid, image_path, size, batch, transform=True, return_valid=True):
    n_train = 10000
    n_valid = 5000
    train = pd.read_csv(train)
    valid = pd.read_csv(valid)
    if not return_valid: 
        train = train.append(valid).sample(frac=1.).reset_index(drop=True)
        n_train = 15000
    valid['weight'] = valid['label'] * 0.8 + 0.01
    train = train.sample(n=n_train, weights=train.weight).reset_index(drop=True)
    valid = valid.sample(n=n_valid, weights=valid.weight).reset_index(drop=True)
    print('Train Data:', round(train.label.mean(),4))
    print('Valid Data:', round(valid.label.mean(),4))
    train = PairTrainDataset(image_path, train,size, transform)
    valid = PairTrainDataset(image_path, valid, size, False)
    train = DataLoader(train, batch_size=batch, shuffle=True, num_workers=16)
    valid = DataLoader(valid, batch_size=batch, shuffle=False, num_workers=16)
    return train, valid

def score_loader(data, train_path, score_path, size, batch):
    data = pd.read_csv(data)
    data = PairScoreDataset(train_path, score_path, data, size, False)
    data = DataLoader(data, batch_size=batch, shuffle=False, num_workers=8)
    return data