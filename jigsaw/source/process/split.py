import pandas as pd
import numpy as np


FEATURES = ['id','comment_text','weight']
LABELS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

data = pd.read_csv('../../data/train.csv')
weights = pd.read_csv('../../data/weights.csv', usecols=['id','weight'])
data = weights.merge(data, on='id')
data = data[FEATURES + LABELS]
data[LABELS] = (data[LABELS].fillna(0).values>=0.5).astype(int)
data['comment_text'] = data['comment_text'].fillna('none blank')
train = data.iloc[:-100000,:]
valid = data.iloc[-100000:,:]

train.to_csv('../../data/train_data.csv', index=False)
valid.to_csv('../../data/valid_data.csv', index=False)

data = pd.read_csv('../../data/test.csv')
data['comment_text'] = data['comment_text'].fillna('none blank')
data.to_csv('../../data/test_data.csv', index=False)