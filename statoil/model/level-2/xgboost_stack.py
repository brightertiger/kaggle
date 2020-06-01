import pandas as pd
import numpy as np
import xgboost as xgb

def summary(data, start):
    data['mean'] = data.iloc[:,start:start+6].mean(axis=1)
    data['median'] = data.iloc[:,start:start+6].median(axis=1)
    data['min'] = data.iloc[:,start:start+6].min(axis=1)
    data['max'] = data.iloc[:,start:start+6].max(axis=1)
    data['diff'] = data['mean'] - data['min']
    return data

train_score = pd.read_csv('../data/train_scores.csv')
train_score = summary(train_score, 2).drop('stack', axis=1)
train_image = pd.read_csv('../data/train_xgb.csv')
train_data = train_score.merge(train_image, on='id')
print(train_data.shape)
train_matrix = xgb.DMatrix(data=train_data.iloc[:,2:], label=train_data['label'])

test_score = pd.read_csv('../data/test_scores.csv')
test_score = summary(test_score, 2).drop('stack', axis=1)
test_image = pd.read_csv('../data/test_xgb.csv')
test_data = test_score.merge(test_image, on='id')
print(test_data.shape)
test_matrix = xgb.DMatrix(data=test_data.iloc[:,1:])

booster = {}
booster['booster'] = 'gbtree'
booster['nthread'] = 6
booster['max_depth'] = 5
booster['min_child_weight'] = 4
booster['subsample'] = 0.8
booster['colsample_bytree'] = 1.0
booster['colsample_bylevel'] = 0.8
booster['lambda'] = 4.0
booster['alpha'] = 3.0
booster['objective'] = 'binary:logistic'
booster['eval_metric'] = ['logloss']
booster['seed'] = 2017
booster['eta'] = 0.01

params = {}
params['params'] = booster
params['dtrain'] = train_matrix
params['num_boost_round'] = 3000
params['early_stopping_rounds'] = 200
params['verbose_eval'] = 400
model = xgb.cv(**params, nfold=5)

params = {}
params['params'] = booster
params['dtrain'] = train_matrix
params['num_boost_round'] = 2400
model = xgb.train(**params)
sorted(model.get_fscore().items(), key=lambda x : x[1], reverse=True)[:30]
scores = model.predict(test_matrix)

submit = test_data[['id']].copy()
submit['is_iceberg'] = scores
submit.to_csv('../data/xgb_stack.csv', index=False)
print(submit.shape)