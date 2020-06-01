import pandas as pd
import numpy as np
import xgboost as xgb

dependent = pd.read_csv('../data/model/dependent/dependent_n.csv')
independent = pd.read_csv('../data/model/independent/independent_n.csv')
print(dependent.shape, independent.shape)
data = dependent.merge(independent, on=['user_id','product_id','eval_set'], how='inner')
del dependent, independent
train_data = data[data['eval_set'] == 'train']
valid_data = data[data['eval_set'] == 'valid']
del data
print(train_data.shape, valid_data.shape)

train_matrix = xgb.DMatrix(data = train_data.iloc[:,4:], label = train_data.iloc[:,3])
del train_data
valid_matrix = xgb.DMatrix(data = valid_data.iloc[:,4:], label = valid_data.iloc[:,3])
del valid_data

booster = {}
booster['booster'] = 'gbtree'
booster['nthread'] = 63
booster['max_depth'] = 10
booster['min_child_weight'] = 10
booster['subsample'] = 0.8
booster['colsample_bytree'] = 1.0
booster['colsample_bylevel'] = 0.9
booster['lambda'] = 1.0
booster['alpha'] = 0.0
booster['objective'] = 'binary:logistic'
booster['eval_metric'] = ['logloss']
booster['base_score'] = 0.1
booster['seed'] = 108

params = {}
params['params'] = booster
params['dtrain'] = train_matrix
params['num_boost_round'] = 2000
params['evals'] = [(train_matrix,'train_matrix'),(valid_matrix,'valid_matrix')]
params['early_stopping_rounds'] = 10
params['verbose_eval'] = 150
params['callbacks'] = [xgb.callback.reset_learning_rate([0.02] * 2000)]

model = xgb.train(**params)
model.save_model('../data/model/xgb_binary.model')
sorted(model.get_fscore().items(), key=lambda x : x[1], reverse=True)[:50]
model.attributes()