import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

FEATURES = ['id','comment_text','weight']
LABELS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

data = pd.read_csv('../../data/train.csv')
weights = pd.read_csv('../../data/weights.csv', usecols=['id','weight'])
data = weights.merge(data, on='id')
data = data[FEATURES + LABELS]
data[LABELS] = (data[LABELS].fillna(0).values>=0.5).astype(int)
data['comment_text'] = data['comment_text'].fillna('none blank')
data = data.reset_index(drop=True)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)
counter = 0
for train_idx, valid_idx in folds.split(data.index, data['target']):
    counter += 1
    train = data.iloc[train_idx,:]
    valid = data.iloc[valid_idx,:]
    train.to_csv('../../data/train_data_{}.csv'.format(counter), index=False)
    valid.to_csv('../../data/valid_data_{}.csv'.format(counter), index=False)

data = pd.read_csv('../../data/test.csv')
data['comment_text'] = data['comment_text'].fillna('none blank')
data.to_csv('../../data/test_data.csv', index=False)