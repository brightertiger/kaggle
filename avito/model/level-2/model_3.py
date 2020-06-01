import pandas as pd
import numpy as np
import lightgbm as lgb

booster = {}
booster['boosting_type'] = 'gbdt'
booster['objective'] = 'regression'
booster['learning_rate'] = 0.01
booster['num_leaves'] = 300
booster['max_depth'] = -1
booster['min_child_weight'] = 50
booster['max_bin'] = 1024
booster['subsample'] = 0.7
booster['subsample_freq'] = 1
booster['colsample_bytree'] = 0.6
booster['min_split_gain'] = 0
booster['nthread'] = 12
booster['verbose'] = 0
booster['metric'] = 'root_mean_squared_error'

def importance(model, feats):
    importance = model.feature_importance(importance_type='gain')
    importance = pd.DataFrame(importance, columns=['importance'])
    importance['feature'] = feats
    importance['importance'] = importance['importance'] / importance['importance'].max()
    importance = importance[['feature', 'importance']]
    importance = importance.sort_values(by='importance', ascending=False)
    importance.to_csv('../data/data/lightgbm/importance_3.csv', index=False)
    return None
    
def execute(mode, imp=False):
    train_data = pd.read_csv('../data/data/lightgbm/dataset/train_3_{}.csv'.format(mode))
    valid_data = pd.read_csv('../data/data/lightgbm/dataset/valid_3_{}.csv'.format(mode))
    train_labels = train_data['deal_probability'].copy()
    valid_labels = valid_data['deal_probability'].copy()
    train_data = train_data.drop(['item_id','deal_probability'], axis=1)
    valid_data = valid_data.drop(['item_id','deal_probability'], axis=1)
    params = {}
    params['label'] = train_labels
    params['feature_name'] = list(train_data.columns)
    train_matrix = lgb.Dataset(train_data.values, **params)
    params = {}
    params['label'] = valid_labels
    params['feature_name'] = list(valid_data.columns)
    valid_matrix = lgb.Dataset(valid_data.values, **params)
    global booster
    params = {}
    params['params'] = booster
    params['train_set'] = train_matrix
    params['valid_sets'] = [train_matrix, valid_matrix]
    params['num_boost_round'] = 5000
    params['early_stopping_rounds'] = 50
    params['verbose_eval'] = 250
    model = lgb.train(**params)
    model.save_model('../data/data/lightgbm/model_3/lightgbm_{}.model'.format(mode))
    if imp:
        importance(model, list(valid_data.columns))
    return None

execute(1, True)
execute(2)
execute(3)
execute(4)
execute(5)