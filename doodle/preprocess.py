import pandas as pd
import numpy as np
import os
import glob
import pickle
from ast import literal_eval
from sklearn.model_selection import StratifiedShuffleSplit
from functools import reduce

FILES_PATH = '../data/download/'
TRAIN_PATH = '../data/train/'
VALID_PATH = '../data/valid/'

# Delete files

categories = glob.glob(FILES_PATH + '/*')
categories = [x.split('/')[-1] for x in categories]
pickle.dump(categories, open('../data/categories.pkl','wb'))

def clean_directory(path):
    files = glob.glob(path + '/*')
    print('deleting {} files...'.format(len(files)))
    for f in files:
        os.remove(f)
    return None

clean_directory(TRAIN_PATH)
clean_directory(VALID_PATH)

# Append all datasets

full_data = pd.DataFrame([])
counter = 0
def append(source):
    global full_data, counter
    data_file = pd.read_csv(FILES_PATH + source)
    # data_file = data_file[data_file['recognized'] == True]
    data_file = data_file[['key_id','drawing','word']]
    full_data = full_data.append(data_file)
    counter += 1
    if counter % 10 == 0:
        print('finished appending {} files'.format(counter))
        print('data shape:', full_data.shape)
    full_data = full_data.reset_index(drop=True)
    return None

for source in categories:
    append(source)

# Train-Test Split

split = StratifiedShuffleSplit(n_splits=1, random_state=2017, test_size=0.1)

for train_idx, valid_idx in split.split(full_data, full_data['word']):
    train_data = full_data.iloc[train_idx,:].reset_index(drop=True)
    train_data = train_data.sample(frac=1, random_state=2017) 
    train_data = train_data.reset_index(drop=True)
    valid_data = full_data.iloc[valid_idx,:].reset_index(drop=True)
    full_data = None
    train_data.to_csv(TRAIN_PATH + 'train.csv', index=False) # 44,736,821
    print('train subset data shape:', train_data.shape)
    valid_data.to_csv(VALID_PATH + 'valid.csv', index=False) # 4,970,758
    print('valid subset data shape:', valid_data.shape)