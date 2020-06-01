import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

model = xgb.Booster({'nthread':7})
model.load_model('../../data/spooky-author/data/xgb_model')

text_feats = pd.read_csv('../../data/spooky-author/data/test_text_feats.csv')
nb_score = pd.read_csv('../../data/spooky-author/data/test_nb_score.csv').drop(['id'], axis=1)
nb_feats = pd.read_csv('../../data/spooky-author/data/test_nb_feats.csv')
nn_score = pd.read_csv('../../data/spooky-author/data/test_nn_score.csv')[['id','keras']]
test_data = text_feats.join(nb_feats)
test_data = test_data.join(nb_score)
test_data = test_data.merge(nn_score, on='id')
print('data shapes:', test_data.shape)
test_matrix = xgb.DMatrix(data = test_data.iloc[:,1:])

score = pd.DataFrame(model.predict(test_matrix), columns = ['EAP','HPL','MWS'])
score = test_data[['id']].join(score)
print(score.shape)
score.to_csv('../../data/spooky-author/data/xgb_score.csv', index=False)