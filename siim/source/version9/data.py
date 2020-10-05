import torch
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2

HTRANS = A.Compose([A.ShiftScaleRotate(rotate_limit=[-45,45],scale_limit=[-0.1,0.1], \
                                           shift_limit=[-0.1,0.15],border_mode=3,value=0,p=1.)])
HAIRS = np.load('../../data/hair_array.npy')

def hairMask(p=0.1):
    chance = np.random.uniform(0,1,1)
    if chance <= p:
        mask_to_chose = np.random.choice(np.arange(7), 1, p=[0.2,0.2,0.22,0.15,0.14,0.06,0.03])[0]
        mask = HAIRS[mask_to_chose]
        mask = HTRANS(image = mask)['image']
        mask = cv2.resize(mask/255,(512,512), cv2.INTER_CUBIC)
        mask[mask == 1.] =  255
        mask[mask != 255.] = 0
    else:
        mask = np.ones((512,512))
    return mask.astype(np.uint8)

def trainTF():
    transforms = []
    transforms += [A.RandomSizedCrop(min_max_height=(400, 500), height=512, width=512, p=0.5)]
    transforms += [A.HorizontalFlip(p=0.5)]
    transforms += [A.VerticalFlip(p=0.5)]
    transforms += [A.RandomRotate90(p=0.5)]
    transforms += [A.RandomBrightnessContrast(p=0.5)]
    transforms += [A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.5)]
    transforms += [A.Resize(height=384, width=384, p=1.)]
    transforms += [A.Normalize()]
    transforms += [ToTensorV2()]
    transforms = A.Compose(transforms, p=1.)
    return transforms

def validTF():
    transforms = []
    transforms += [A.HorizontalFlip(p=0.5)]
    transforms += [A.VerticalFlip(p=0.5)]
    transforms += [A.Resize(height=384, width=384, p=1.)]
    transforms += [A.Normalize()]
    transforms += [ToTensorV2()]
    transforms = A.Compose(transforms, p=1.)
    return transforms

LOOKUP = {'other' : 0, 'melanoma' : 1, 'nevus' : 2, 'keratosis' : 3}

class TrainDataset(Dataset):

    def __init__(self, path, data, fold):
        self.path = path
        self.data = data[(data['fold'] != fold) & (data['source'] != 'IGNORE')]
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask =  hairMask()
        image = cv2.bitwise_and(image, image, mask=mask)
        image = self.transform(image=image)['image']
        return image

    def __label__(self, idx):
        label = self.data.loc[idx,'diagnosis']
        array = [0.,0.,0.,0.]
        array[LOOKUP[label]] += 1.
        array = np.array(array).astype(float).reshape(4,)
        return array
    
    def __getitem__(self, idx):
        image = self.__image__(idx)
        label = self.__label__(idx)
        sample = {'image': image, 'label': label}
        return sample

class ValidDataset(Dataset):

    def __init__(self, path, data, fold):
        self.path = path
        self.transform = validTF()
        self.data = data[(data['fold'] == fold) & (data['source'] == 'ISIC20')]
        self.data = self.data.reset_index(drop=True)
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

    def __label__(self, idx):
        label = self.data.loc[idx,'diagnosis']
        array = [0.,0.,0.,0.]
        array[LOOKUP[label]] += 1.
        array = np.array(array).astype(float).reshape(4,)
        return array
    
    def __getitem__(self, idx):
        image = self.__image__(idx)
        label = self.__label__(idx)
        sample = {'image': image, 'label': label}
        return sample   
    
def trainLoader(image_path, label_path, fold_idx):
    data = pd.read_csv(label_path)
    train = TrainDataset(image_path, data, fold_idx)
    valid = ValidDataset(image_path, data, fold_idx)
    print('Train Images:', len(train), 'Valid Images:', len(valid))
    return train, valid