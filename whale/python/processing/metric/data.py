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

class ImageDataset(Dataset):

    def __init__(self, path, data, size, transform = True):
        self.path = path
        self.size = (size,size)
        if transform : self.transform = TRAIN_TFMS 
        else : self.transform = VALID_TFMS 
        data = data.reset_index(drop=True)
        self.image = data['Image'].tolist()
        self.label = data['Id'].tolist()
        return None

    def __len__(self):
        return len(self.image)

    def __image__(self, image_idx):
        image = Image.open(self.path + '/' + image_idx).convert('RGB')
        image = image.resize(self.size, resample=PIL.Image.BICUBIC)
        image = self.transform(image=np.array(image))
        image = image['image']
        image = image.transpose(2,0,1)
        return image

    def __getitem__(self, idx):
        image = self.__image__(self.image[idx])
        label = self.label[idx]
        label = np.array(label).reshape(1, 1, 1)
        label = torch.from_numpy(label)
        image = torch.from_numpy(image)
        sample = {'idx':self.image[idx], 'image': image, 'label': label}
        return sample

def pretrain_loader(image_path, label_path, size, batch, use_weight=False):
    data = pd.read_csv(label_path)
    data['count'] = 1 / (2 + data['count'])
    if not use_weight:
        sampler = None
        dataset = ImageDataset(image_path, data, size, True)
        dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=16)
    else:
        sampler = data['count'].tolist()
        sampler = WeightedRandomSampler(sampler, len(sampler))
        dataset = ImageDataset(image_path, data, size, True)
        dataloader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=16, sampler=sampler)
    return dataloader

def data_loader(image_path, label_path, size, batch, return_valid=True, use_weight=False):
    encoder = LabelEncoder()
    data = pd.read_csv(label_path)
    data = data[data['Id'] != 'new_whale'].reset_index(drop=True)
    data['Id'] = encoder.fit_transform(data['Id'])
    count = data.Id.value_counts()
    count.name = 'count'
    data = data.join(count, on='Id')
    valid = set(data[(data['count'] > 1)].groupby('Id').first().Image)
    train = data[np.logical_not(data['Image'].isin(valid))].reset_index(drop=True)
    valid = data[data['Image'].isin(valid)].reset_index(drop=True)
    train['count'] = 1 / (2 + train['count'])
    valid['count'] = 1 / (2 + valid['count'])
    if not return_valid: train = train.append(valid).reset_index(drop=True)
    if not use_weight:
        train = ImageDataset(image_path, train, size, True)
        valid = ImageDataset(image_path, valid, size, False)
        print('Train Images:', len(train), 'Valid Images:', len(valid))
        train = DataLoader(train, batch_size=batch, shuffle=True, num_workers=16)
        valid = DataLoader(valid, batch_size=1, shuffle=False, num_workers=16)
    else:
        sampler = train['count'].tolist()
        sampler = WeightedRandomSampler(sampler, len(sampler))
        train = ImageDataset(image_path, train, size, True)
        valid = ImageDataset(image_path, valid, size, False)
        print('Train Images:', len(train), 'Valid Images:', len(valid))
        train = DataLoader(train, batch_size=batch, shuffle=False, num_workers=16, sampler=sampler)
        valid = DataLoader(valid, batch_size=1, shuffle=False, num_workers=16)
    return train, valid

def score_loader(image_path, size, batch, transform=False):
    files = glob.glob(image_path + '/*jpg')
    files = [x.split('/')[-1] for x in files]
    data = pd.DataFrame([])
    data['Image'] = files
    data['Id'] = [0] * data.shape[0]
    print('Score Images:', data.shape[0])
    data = ImageDataset(image_path, data, size, transform)
    data = DataLoader(data, batch_size=batch, shuffle=False, num_workers=16)
    return data

