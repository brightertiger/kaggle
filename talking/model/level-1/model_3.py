import pandas as pd
import numpy as np
import lightgbm as lgb

categorical = ['ip','app','channel','device','os']

train_data = pd.read_feather('../data/data/model/full_data.feather', nthreads=15)

valid_ids = pd.read_feather('../data/data/model/valid_data.feather', nthreads=15)[['click_id']]
train_ids = set(train_data['click_id']) - set(valid_ids['click_id'])
train_ids = pd.DataFrame(list(train_ids), columns=['click_id'])
print(len(train_ids), len(valid_ids))

valid_data = train_data.merge(valid_ids, on='click_id')
valid_data.to_feather('../data/data/model/full_data/valid_data.feather')
del valid_data

train_data = train_data.merge(train_ids, on='click_id')
train_data.to_feather('../data/data/model/full_data/train_data.feather')
del train_data

train_data = pd.read_feather('../data/data/model/full_data/train_data.feather', nthreads=15)
train_labels = train_data['is_attributed'].values
train_data = train_data.drop(['is_attributed','day','click_id'], axis=1)
train_weights = train_data['hour'].map(lambda x : 2 if x in [4,5,9,10,13,14] else 1).astype('uint8')

params = {}
params['label'] = train_labels
params['feature_name'] = list(train_data.columns)
params['categorical_feature'] = categorical
params['weight'] = train_weights
train_matrix = lgb.Dataset(np.array(train_data.values, dtype=np.float32), **params)
del train_data

train_matrix.save_binary('../data/data/model/full_data/train_matrix.bin')
train_matrix = lgb.Dataset('../data/data/model/full_data/train_matrix.bin')

valid_data = pd.read_feather('../data/data/model/full_data/valid_data.feather', nthreads=15)
valid_labels = valid_data['is_attributed'].values
valid_data = valid_data.drop(['is_attributed','day','click_id'], axis=1)

params = {}
params['label'] = valid_labels
params['feature_name'] = list(valid_data.columns)
params['categorical_feature'] = categorical
valid_matrix = lgb.Dataset(valid_data.values, **params)

booster = {}
booster['boosting_type'] = 'gbdt'
booster['objective'] = 'binary'
booster['learning_rate'] = 0.1
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
params['train_set'] = train_matrix
params['valid_sets'] = [valid_matrix]
params['num_boost_round'] = 1000
params['early_stopping_rounds'] = 50
params['verbose_eval'] = 50

model = lgb.train(**params)
model.save_model('../data/data/model/lightgbm_3.model')

train_data = pd.read_feather('../data/data/model/full_data.feather', nthreads=15)
train_labels = train_data['is_attributed'].values
train_data = train_data.drop(['is_attributed','day','click_id'], axis=1)
train_weights = train_data['hour'].map(lambda x : 2 if x in [4,5,9,10,13,14] else 1).astype('uint8')

params = {}
params['label'] = train_labels
params['feature_name'] = list(train_data.columns)
params['categorical_feature'] = categorical
params['weight'] = train_weights
train_matrix = lgb.Dataset(np.array(train_data.values, dtype=np.float32), **params)
del train_data

train_matrix.save_binary('../data/data/model/full_data/full_matrix.bin')
del train_matrix
train_matrix = lgb.Dataset('../data/data/model/full_data/full_matrix.bin')

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
params['num_boost_round'] = 375

model = lgb.train(**params)

model.save_model('../data/data/model/lightgbm_4.model')