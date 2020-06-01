import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc

valid_data = pd.read_feather('../data/data/model/valid_data.feather')
valid_labels = valid_data['is_attributed'].values
valid_driver = valid_data[['click_id', 'is_attributed']].copy()
valid_data = valid_data.drop(['is_attributed','day','click_id'], axis=1)

score_data = pd.read_feather('../data/data/model/score_data.feather')
score_driver = score_data[['click_id']].copy()
score_data = score_data.drop(['day','click_id'], axis=1)

model = lgb.Booster(model_file='../data/data/model/lightgbm_2.model')

valid_driver['score'] = model.predict(valid_data)
score_driver['is_attributed'] = model.predict(score_data)
valid_driver.to_csv('../data/score/lightgbm_6.csv', index=False)
score_driver.to_csv('../data/submit/lightgbm_6.csv', index=False)
fpr, tpr, thresh = roc_curve(valid_driver['is_attributed'], valid_driver['score'])
auc(fpr,tpr)