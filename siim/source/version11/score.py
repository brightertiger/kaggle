import sys
sys.path.insert(0,'..')
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import albumentations as A
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
from albumentations.pytorch.transforms import ToTensorV2
from collections import defaultdict

DEVICE = 'cuda:0'
SEX_LOOKUP = defaultdict(int,{'male':1, 'female':2})
AGE_LOOKUP = defaultdict(int,{20:1, 30:2, 40:3, 50:4, 60:5, 70:6, 80:7})
ANT_LOOKUP = defaultdict(int,{'lower extremity':1, 'upper extremity':2, 'torso':3, 'head/neck':4})

def scoreTF():
    transforms = []
    transforms += [A.HorizontalFlip(p=0.5)]
    transforms += [A.VerticalFlip(p=0.5)]
    transforms += [A.Resize(height=512, width=512, p=1.0)]
    transforms += [A.Normalize()]
    transforms += [ToTensorV2()]
    transforms = A.Compose(transforms, p=1.)
    return transforms

class ScoreDataset(Dataset):

    def __init__(self, path, data):
        self.path = path
        self.data = data.reset_index(drop=True).copy()
        self.data['age_approx'] = self.data.age_approx.round(-1).clip(20,80)
        self.transform = scoreTF()
        return None
    
    def __len__(self):
        return len(self.data)

    def __image__(self, idx):
        image = self.data.loc[idx, 'image_id']
        name = self.path + '/' + image + '.jpg'
        image = cv2.imread(name, cv2.IMREAD_COLOR)
        image = self.transform(image=image)['image']
        return image
    
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
        metad = self.__metad__(idx)
        sample = {'image': image, 'metad': metad}
        return sample

def scoreModel(model, data):
    model.eval()
    pred_array = []
    with torch.no_grad():
        for sample in data:
            image = Variable(sample['image'].float().to(DEVICE))
            metad = Variable(sample['metad'].float().to(DEVICE))
            preds = torch.sigmoid(model(image, metad).squeeze())
            pred_array.append(preds.cpu().data.numpy())
    pred_array = np.vstack(pred_array)[:,1]
    return pred_array