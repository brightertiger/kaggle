import pandas as pd
import numpy as np
import cv2
import pickle
import os
import shutil
from sklearn.model_selection import StratifiedKFold

INPATH = '../data/download/'
OUTPATH = '../data/data/'

def image_category(file):
    mask = cv2.imread(INPATH + 'train/masks/{}.png'.format(file))
    cover = (mask>0.5).sum()
    percentage = cover/(101*101)
    if cover < 8:
        return 0
    if np.all(mask==mask[0]):
        return 1
    if percentage <= 0.15:
        return 2
    if percentage <= 0.25:
        return 3
    if percentage <= 0.50:
        return 4
    if percentage <= 0.67:
        return 5
    else:
        return 6  

files = pd.read_csv(INPATH + 'train.csv')
types = files['id'].map(lambda x : image_category(x))
files = files.id.values

folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=2017)
fold = 1

for train_idx, valid_idx in folds.split(files, types):
    train_path = OUTPATH + 'fold_{}/train/'.format(fold)
    valid_path = OUTPATH + 'fold_{}/valid/'.format(fold)
    train_images = files[train_idx]
    valid_images = files[valid_idx]
    print('# images:', len(train_images), len(valid_images))
    for file in train_images:
        image = cv2.imread(INPATH + 'train/images/{}.png'.format(file))
        image = cv2.resize(image,(101,101))
        cv2.imwrite(train_path + '/images/{}.png'.format(file), image)
        mask = cv2.imread(INPATH + 'train/masks/{}.png'.format(file))
        mask = cv2.resize(mask,(101,101))
        cv2.imwrite(train_path + '/masks/{}.png'.format(file), mask)
    with open(train_path + 'train.pkl', 'wb') as file:
        pickle.dump(train_images, file)

    for file in valid_images:
        image = cv2.imread(INPATH + 'train/images/{}.png'.format(file))
        image = cv2.resize(image,(101,101))
        cv2.imwrite(valid_path + '/images/{}.png'.format(file), image)
        mask = cv2.imread(INPATH + 'train/masks/{}.png'.format(file))
        mask = cv2.resize(mask,(101,101))
        cv2.imwrite(valid_path + '/masks/{}.png'.format(file), mask)
    with open(valid_path + 'valid.pkl', 'wb') as file:
        pickle.dump(valid_images, file)
    fold += 1

test = pd.read_csv(INPATH + '/depths.csv')
test = list(sorted(list(set(test.id.values) - set(files))))
print('# images:', len(test))

for file in test:
    image = cv2.imread(INPATH + '/test/images/{}.png'.format(file))
    image = cv2.resize(image,(101,101))
    cv2.imwrite(OUTPATH + '/test/images/{}.png'.format(file), image)

with open(OUTPATH + 'test/test.pkl', 'wb') as file:
    pickle.dump(test, file)