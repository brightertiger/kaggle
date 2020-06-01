import pandas as pd
import numpy as np

data = pd.read_csv('../../data/train.csv')

subset = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish']
subset += ['muslim', 'black', 'white', 'psychiatric_or_mental_illness']

weights = data[['id','target'] + subset].copy()
weights['base'] = 1
weights['normal'] = (weights[subset].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(int)
weights['group_1'] = (weights['target'].values>=0.5).astype(bool).astype(np.int)
weights['group_1'] += (weights[subset].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(int)
weights['group_1'] = (weights['group_1'] > 1).astype(bool).astype(int)
weights['group_2'] = (weights['target'].values<0.5).astype(bool).astype(np.int)
weights['group_2'] += (weights[subset].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(int)
weights['group_2'] = (weights['group_2'] > 1).astype(bool).astype(int)
weights = weights[['id', 'base', 'normal','group_1','group_2']]
weights['weight'] = weights['base'] + weights['normal'] + weights['group_1'] + weights['group_2']
weights['weight'] = weights['weight'] / 4
weights.to_csv('../../data/weights.csv', index=False)

