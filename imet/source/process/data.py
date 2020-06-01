import torch
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

class ImageDataset(Dataset):

    def __init__(self, path, data, size, transform=True):
        self.path = path
        self.size = size
        if transform : self.transform = self.__train__()
        else : self.transform = self.__score__()
        data = data.reset_index(drop=True)
        self.image = data['id'].tolist()
        self.label = data['attribute_ids'].tolist()
        return None
    
    def __train__(self):
        transform = []
        transform.append(transforms.RandomHorizontalFlip(p=0.5))
        transform.append(transforms.RandomCrop(size=(self.size,self.size), pad_if_needed=True))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform)
        return transform
    
    def __score__(self):
        transform = []
        transform.append(transforms.RandomCrop(size=(self.size,self.size), pad_if_needed=True))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform)
        return transform

    def __len__(self):
        return len(self.image)

    def __image__(self, image_idx):
        image = Image.open(self.path + '/' + image_idx + '.png')
        image = self.transform(image)
        return image

    def __label__(self, label_idx):
        label = self.label[label_idx]
        label = label.split()
        epsilon = 0.1
        array = [epsilon / 1000] * 1103
        for idx in label:
            array[int(idx)] = 1 - epsilon
        array = np.array(array)
        array = torch.from_numpy(array).reshape(1,-1)
        return array

    def __getitem__(self, idx):
        image = self.__image__(self.image[idx])
        label = self.__label__(idx)
        sample = {'idx': self.image[idx], 'image': image, 'label': label}
        return sample

def trainLoader(image_path, label_path, size, batch, fold_idx):
    data = pd.read_csv(label_path)
    train = data[data['fold'] != fold_idx].reset_index(drop=True)
    valid = data[data['fold'] == fold_idx].reset_index(drop=True)
    train = ImageDataset(image_path, train, size, True)
    valid = ImageDataset(image_path, valid, size, False)
    print('Train Images:', len(train), 'Valid Images:', len(valid))
    train = DataLoader(train, batch_size=batch, shuffle=True, num_workers=6, drop_last=True)
    valid = DataLoader(valid, batch_size=max(batch//2,1), shuffle=False, num_workers=6, drop_last=True)
    return train, valid

def scoreLoader(image_path, label_path, size, batch, fold_idx):
    data = pd.read_csv(label_path)
    valid = data[data['fold'] == fold_idx].reset_index(drop=True)
    valid = ImageDataset(image_path, valid, size, False)
    print('Score Images:', len(valid))
    valid = DataLoader(valid, batch_size=batch, shuffle=False, num_workers=4, drop_last=True)
    return valid
