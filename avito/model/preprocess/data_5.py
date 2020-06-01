import pandas as pd
import numpy as np
import string
import re
from sklearn.preprocessing import LabelEncoder

idx = ['item_id', 'image']
encode = ['parent_category_name','category_name', 'image_top_1']

def label_encode(variable):
    global train_data, test_data, full_data
    encode = LabelEncoder()
    encode.fit(full_data[variable].astype(str).fillna('none'))
    train_data[variable] = encode.transform(train_data[variable].astype(str).fillna('none'))
    test_data[variable] = encode.transform(test_data[variable].astype(str).fillna('none'))
    return None

train_data = pd.read_csv('../data/download/train.csv', usecols = encode + idx +['deal_probability'])
test_data = pd.read_csv('../data/download/test.csv', usecols = encode + idx)
full_data = train_data.append(test_data)

label_encode('parent_category_name')
label_encode('category_name')
label_encode('image_top_1')

train_data = train_data[idx + ['deal_probability'] + encode]
test_data = test_data[idx + encode]

for feature in encode:
    maximum = max(train_data[feature].max(), test_data[feature].max())
    print('dimension:', feature, maximum + 1)

train_data = train_data[np.logical_not(train_data['image'].isnull())]
test_data = test_data[np.logical_not(test_data['image'].isnull())]

exclude = []
exclude += ['b98b291bd04c3d92165ca515e00468fd9756af9a8f1df42505deed1dcfb5d7ae']
exclude += ['8513a91e55670c709069b5f85e12a59095b802877715903abef16b7a6f306e58']
exclude += ['60d310a42e87cdf799afcd89dc1b11ae3fdc3d0233747ec7ef78d82c87002e83']
exclude += ['4f029e2a00e892aa2cac27d98b52ef8b13d91471f613c8d3c38e3f29d4da0b0c']

train_data = train_data[np.logical_not(train_data['image'].isin(exclude))]
train_data.to_csv('../data/data/image/train_data.csv', index=False)
test_data.to_csv('../data/data/image/test_data.csv', index=False)

for i in range(5):
    i += 1
    train_idx = pd.read_csv('../data/data/files/train_{}.csv'.format(i))
    valid_idx = pd.read_csv('../data/data/files/valid_{}.csv'.format(i))
    train_sub = train_idx.merge(train_data, on='item_id')
    valid_sub = valid_idx.merge(train_data, on='item_id')
    print('dataset:', train_sub.shape, valid_sub.shape)
    train_sub.to_csv('../data/data/image/dataset/train_{}.csv'.format(i), index=False)
    valid_sub.to_csv('../data/data/image/dataset/valid_{}.csv'.format(i), index=False)