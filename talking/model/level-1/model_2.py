import pandas as pd
import numpy as np
import lightgbm as lgb

categorical = ['ip','app','channel','device','os']

train_data = pd.read_feather('../data/data/model/train_data.feather', nthreads=15)
valid_data = pd.read_feather('../data/data/model/valid_data.feather', nthreads=15)
data = train_data.append(valid_data)
train_data, valid_data = None, None
labels = data['is_attributed'].values
data = data.drop(['is_attributed','day','click_id'], axis=1)
matrix = lgb.Dataset(data.values, label=labels,feature_name=list(data.columns), categorical_feature=categorical)

booster = {}
booster['boosting_type'] = 'gbdt'
booster['objective'] = 'binary'
booster['learning_rate'] = 0.075
booster['num_leaves'] = 32
booster['max_depth'] = -1
booster['min_child_weight'] = 5
booster['max_bin'] = 255
booster['subsample'] = 0.6
booster['subsample_freq'] = 1
booster['colsample_bytree'] = 0.3
booster['min_split_gain'] = 0
booster['nthread'] = 15
booster['verbose'] = 0
booster['scale_pos_weight'] = 99.7
booster['metric'] = 'auc'

params = {}
params['params'] = booster
params['train_set'] = matrix
params['num_boost_round'] = 450
params['verbose_eval'] = 100

model = lgb.train(**params)

model.save_model('../data/data/model/lightgbm_1.model')

matrix = lgb.Dataset(data.values, label=labels,feature_name=list(data.columns), categorical_feature=categorical)

booster = {}
booster['boosting_type'] = 'gbdt'
booster['objective'] = 'binary'
booster['learning_rate'] = 0.1
booster['num_leaves'] = 24
booster['max_depth'] = -1
booster['min_child_weight'] = 5
booster['max_bin'] = 255
booster['subsample'] = 0.5
booster['subsample_freq'] = 1
booster['colsample_bytree'] = 0.3
booster['min_split_gain'] = 0
booster['nthread'] = 15
booster['verbose'] = 0
booster['scale_pos_weight'] = 99.7
booster['metric'] = 'auc'

params = {}
params['params'] = booster
params['train_set'] = matrix
params['num_boost_round'] = 425
params['verbose_eval'] = 100

model = lgb.train(**params)
model.save_model('../data/data/model/lightgbm_2.model')