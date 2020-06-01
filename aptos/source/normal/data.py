import torch
import cv2
import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageStat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

def squareImage(image, size):
    image = image.resize((size,size), Image.ANTIALIAS)
    transform = []
    transform.append(transforms.RandomHorizontalFlip(p=0.5))
    transform.append(transforms.RandomVerticalFlip(p=0.5))
    transform.append(transforms.ColorJitter(0.25, 0.25, 0.25))
    transform.append(transforms.RandomCrop((size,size), pad_if_needed=True))
    transform.append(transforms.RandomAffine(0, translate=None, scale=(1.0,1.25)))
    transform = transforms.Compose(transform)
    image = transform(image)
    return image

def rectangleImage(image, size):
    height, width = image.size
    if height > width:
        aspect = width / height
        new_height = int(size)
        new_width = int(np.rint(aspect * size))
    else:
        aspect = height / width
        new_height = int(np.rint(aspect * size))
        new_width = int(size)
    image = image.resize((new_height, new_width), Image.ANTIALIAS)
    transform = []
    transform.append(transforms.RandomHorizontalFlip(p=0.5))
    transform.append(transforms.RandomVerticalFlip(p=0.5))
    transform.append(transforms.ColorJitter(0.25, 0.25, 0.25))
    transform.append(transforms.RandomCrop((size,size), pad_if_needed=True))
    transform.append(transforms.RandomAffine(0, translate=None, scale=(1.0,1.25)))
    transform = transforms.Compose(transform)
    image = transform(image)
    return image

class ImageDataset(Dataset):

    def __init__(self, path, data, size, weight, noise):
        self.path = path
        self.size = size
        data = data.reset_index(drop=True)
        self.image = data['id_code'].tolist()
        self.label = data['diagnosis'].tolist()
        self.weight = weight
        self.noise = noise
        return None
    
    def __len__(self):
        return len(self.image)

    def __image__(self, image_idx):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        name = self.path + '/' + image_idx + '.jpg'
        image = Image.open(name)
        height, width = image.size
        if abs(height - width) < 20:
            image = squareImage(image, self.size)
        else:
            image = rectangleImage(image, self.size)
        transform = []
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(transform)
        image = transform(image)
        return image

    def __label__(self, label_idx):
        label = self.label[label_idx]
        if self.noise: label = np.random.normal(loc=label, scale=0.05)
        array = np.array([label]).clip(0.,4.)
        array = torch.from_numpy(array).reshape(-1,1)
        return array
    
    def __getitem__(self, idx):
        image = self.__image__(self.image[idx])
        label = self.__label__(idx)
        weight = torch.from_numpy(np.array([self.weight])).reshape(-1,1)
        sample = {'idx': self.image[idx], 'image': image, 'label': label, 'weight': weight}
        return sample
        
def trainLoader(image_path, label_path, size, fold_idx, weight=1.):
    data = pd.read_csv(label_path)
    train = data[data['fold'] != fold_idx].reset_index(drop=True)
    valid = data[data['fold'] == fold_idx].reset_index(drop=True)
    train = ImageDataset(image_path, train, size, weight, noise=True)
    valid = ImageDataset(image_path, valid, size, 1., noise=False)
    print('Train Images:', len(train), 'Valid Images:', len(valid))
    return train, valid
