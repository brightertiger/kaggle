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

DEVICE = 'cuda:0'

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
        self.data = data.reset_index(drop=True)
        self.transform = scoreTF()
        return None
    
    def __len__(self):
        return len(self.data)

    def __image__(self, idx):
        image = self.data.loc[idx, 'image_id']
        name = self.path + '/' + image + '.jpg'
        image = cv2.imread(name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        return image
    
    def __getitem__(self, idx):
        image = self.__image__(idx)
        sample = {'image': image}
        return sample

def scoreModel(model, data):
    model.eval()
    pred_array = []
    with torch.no_grad():
        for sample in data:
            image = Variable(sample['image'].float().to(DEVICE))
            preds = torch.sigmoid(model(image).squeeze())
            pred_array.append(preds.cpu().data.numpy())
    pred_array = np.vstack(pred_array)
    return pred_array