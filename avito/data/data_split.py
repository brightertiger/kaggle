import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
get_ipython().magic('matplotlib inline')

data = pd.read_csv('../data/download/train.csv')
data = data.drop(['image'], axis=1)
data['activation_date'] = pd.to_datetime(data['activation_date'])
print('data:', data.shape)

folds = KFold(n_splits=5, shuffle=True, random_state=2017)

fold = 1

for train_idx, test_idx in folds.split(data):
    train = data.iloc[train_idx, :][['item_id']]
    valid = data.iloc[test_idx, :][['item_id']]
    print('datasets:', train.shape, valid.shape)
    train.to_csv('../data/data/files/train_{}.csv'.format(fold), index=False)
    valid.to_csv('../data/data/files/valid_{}.csv'.format(fold), index=False)
    fold += 1

data = pd.read_csv('../data/download/test.csv')[['item_id']]
data.to_csv('../data/data/files/score.csv', index=False)