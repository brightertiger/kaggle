import pandas as pd
import numpy as np
import lightgbm as lgb

categorical = ['ip','app','channel','device','os']

train_data = pd.read_feather('../data/data/model/train_data.feather', nthreads=15)
train_labels = train_data['is_attributed'].values
train_data = train_data.drop(['is_attributed','day','click_id'], axis=1)

valid_data = pd.read_feather('../data/data/model/valid_data.feather')
valid_labels = valid_data['is_attributed'].values
valid_data = valid_data.drop(['is_attributed','day','click_id'], axis=1)

params = {}
params['label'] = train_labels
params['feature_name'] = list(train_data.columns)
params['categorical_feature'] = categorical
train_matrix = lgb.Dataset(train_data.values, **params)

params = {}
params['label'] = valid_labels
params['feature_name'] = list(valid_data.columns)
params['categorical_feature'] = categorical
valid_matrix = lgb.Dataset(valid_data.values, **params)

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
params['train_set'] = train_matrix
params['valid_sets'] = [valid_matrix]
params['num_boost_round'] = 1000
params['early_stopping_rounds'] = 50
params['verbose_eval'] = 50

model = lgb.train(**params)
model.save_model('../data/data/model/lightgbm_1.model')

params = {}
params['label'] = train_labels
params['feature_name'] = list(train_data.columns)
params['categorical_feature'] = categorical
train_matrix = lgb.Dataset(train_data.values, **params)

params = {}
params['label'] = valid_labels
params['feature_name'] = list(valid_data.columns)
params['categorical_feature'] = categorical
valid_matrix = lgb.Dataset(valid_data.values, **params)

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
params['train_set'] = train_matrix
params['valid_sets'] = [valid_matrix]
params['num_boost_round'] = 1000
params['early_stopping_rounds'] = 50
params['verbose_eval'] = 100

model = lgb.train(**params)
model.save_model('../data/data/model/lightgbm_2.model')

importance = model.feature_importance(importance_type='gain')
importance = pd.DataFrame(importance, columns=['importance'])
importance['feature'] = list(valid_data.columns)
importance['importance'] = importance['importance'] / importance['importance'].max()
importance = importance[['feature', 'importance']]
importance = importance.sort_values(by='importance', ascending=False)
importance.to_csv('../data/data/model/variable_importance.csv', index=False)
importance.reset_index(drop=True)