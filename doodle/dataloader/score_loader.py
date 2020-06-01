import torch
import torchvision
import cv2
import pickle
import numpy as np
import pandas as pd
from ast import literal_eval
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as functional
torch.set_default_tensor_type('torch.DoubleTensor')
np.random.seed(2017)

class ScoreDataset(Dataset):

    def __init__(self, dataset, mapping, size):
        self.dataset = dataset['drawing'].tolist()
        self.index = dataset['key_id'].tolist()
        self.mapping = mapping
        self.size = size
        return None

    def __len__(self):
        return len(self.index)

    def __image__(self, array):
        array = literal_eval(array)
        image = np.zeros((256,256))
        for n, stroke in enumerate(array):
            for i in range(len(stroke[0]) - 1):
                color = 255 - min(n, 10) * 13
                x1 = stroke[0][i]
                y1 = stroke[1][i]
                x2 = stroke[0][i + 1]
                y2 = stroke[1][i + 1]
                _ = cv2.line(image, (x1,y1), (x2,y2), color, 6)
        image = cv2.resize(image,(self.size,self.size))
        image = np.atleast_3d(image)
        image = image - np.zeros_like(image)
        image = (image / 255.).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image
                
    def __getitem__(self, idx):
        image = self.__image__(self.dataset[idx])
        image = torch.from_numpy(image)
        sample = {'image' : image}
        return sample