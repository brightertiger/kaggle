
import torch
import torchvision
import numpy as np
import cv2
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as functional
torch.set_default_tensor_type('torch.DoubleTensor')
np.random.seed(2017)

class ImageDataset(Dataset):

    def __init__(self, idx, root_dir, flip):
        self.idx = idx
        self.root_dir = root_dir
        self.flip = flip
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return None

    def __len__(self):
        return len(self.idx)

    def __transform__(self, image, mask):
        flip = np.random.binomial(1, 0.5, size=None)
        flip = flip if self.flip else 0
        if flip:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        return image, mask

    def __getitem__(self, idx):
        name = self.idx[idx]
        image = self.root_dir + '/images/' + name + '.png'
        mask = self.root_dir + '/masks/' + name + '.png'
        image = cv2.imread(image)
        mask = cv2.imread(mask)
        image = cv2.copyMakeBorder(image, 14, 13, 14, 13, cv2.BORDER_REPLICATE)
        mask = cv2.copyMakeBorder(mask, 14, 13, 14, 13, cv2.BORDER_REPLICATE)
        image = (image / 255.).astype(np.float32)
        mask = (mask / 255.).astype(np.float32)
        image, mask = self.__transform__(image, mask)
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)
        image = image - np.zeros_like(image)
        mask = mask - np.zeros_like(mask)
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        sample = {'image' : image, 'mask' : mask[:,:,0].reshape(1,128,128)}
        return sample

class ScoreDataset(Dataset):

    def __init__(self, idx, root_dir, flip=False):
        self.idx = idx
        self.root_dir = root_dir
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.flip = flip
        return None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        name = self.idx[idx]
        image = self.root_dir + '/images/' + name + '.png'
        image = cv2.imread(image)
        image = cv2.copyMakeBorder(image, 14, 13, 14, 13, cv2.BORDER_REPLICATE)
        if self.flip:
            image = np.fliplr(image)
        image = (image / 255.).astype(np.float32)
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)
        image = image - np.zeros_like(image)
        image = torch.from_numpy(image)
        sample = {'image' : image}
        return sample
        