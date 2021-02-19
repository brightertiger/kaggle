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
    transforms += [A.Resize(500,500, p=1.)]
    transforms += [A.CenterCrop(384, 384, p=1.)]
    transforms += [A.HorizontalFlip(p=0.5)]
    transforms += [A.VerticalFlip(p=0.5)]
    transforms += [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],                                        max_pixel_value=255.0)]
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
        name = self.path + '/' + image
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
            preds = torch.softmax(model(image).squeeze(), dim=1)
            pred_array.append(preds.cpu().data.numpy())
    pred_array = np.vstack(pred_array)
    return pred_array