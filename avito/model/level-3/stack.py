import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

actual = pd.read_csv('../data/download/train.csv', usecols=['item_id','deal_probability'])
actual = actual.rename(columns={'deal_probability':'actual'})
print('data:', actual.shape)

score = pd.read_csv('../data/download/test.csv', usecols=['item_id'])
print('data:', score.shape)

features = []

def merge(file, name):
    global actual, features
    path = '../data/insample/' + file
    file = pd.read_csv(path)
    file = file.rename(columns={'deal_probability':name})
    actual = actual.merge(file, on='item_id')
    features += [name]
    rmse = np.sqrt(mean_squared_error(actual['actual'], actual[name]))
    print('rmse:', round(rmse,5))
    return None

merge('scores/cnn_3.csv', 'model_0')
merge('scores/lightgbm_1.csv', 'model_1')
merge('scores/lightgbm_2.csv', 'model_2')
merge('scores/lightgbm_3.csv', 'model_3')
merge('team/moiz_2204.csv', 'model_4')
merge('team/moiz_2206.csv', 'model_5')
merge('team/moiz_2209.csv', 'model_6')
merge('team/moiz_2211.csv', 'model_7')
merge('team/moiz_2226.csv', 'model_8')
merge('team/moiz_2233.csv', 'model_9')
merge('team/svj_2179.csv', 'model_10')
merge('team/rk_2208.csv', 'model_11')

print('data:', actual.shape)
actual = actual.sort_values(by='item_id').reset_index(drop=True)
actual.head()

def merge(file, name):
    global score
    path = '../data/outsample/' + file
    file = pd.read_csv(path)
    file = file.rename(columns={'deal_probability':name})
    score = score.merge(file, on='item_id')
    return None

merge('scores/cnn_3.csv', 'model_0')
merge('scores/lightgbm_1.csv', 'model_1')
merge('scores/lightgbm_2.csv', 'model_2')
merge('scores/lightgbm_3.csv', 'model_3')
merge('team/moiz_2204.csv', 'model_4')
merge('team/moiz_2206.csv', 'model_5')
merge('team/moiz_2209.csv', 'model_6')
merge('team/moiz_2211.csv', 'model_7')
merge('team/moiz_2226.csv', 'model_8')
merge('team/moiz_2233.csv', 'model_9')
merge('team/svj_2179.csv', 'model_10')
merge('team/rk_2208.csv', 'model_11')

print('data:', score.shape)
score = score.sort_values(by='item_id').reset_index(drop=True)
actual[features].corr()
score[features].corr()

def summary(data, features):
    data['mean'] = data[features].mean(axis=1)
    data['std'] = data[features].std(axis=1)
    data['min'] = data[features].min(axis=1) 
    data['max'] = data[features].max(axis=1)
    data['ratio'] = data['model_11'] / data['mean']
    return data

actual = summary(actual, ['model_3','model_4','model_5','model_6','model_10','model_11'])
score = summary(score, ['model_3','model_4','model_5','model_6','model_10','model_11'])

train_weak = pd.read_csv('../data/data/features/moiz/train_weak.csv')
test_weak = pd.read_csv('../data/data/features/moiz/test_weak.csv')
actual = actual.merge(train_weak, on='item_id')
score = score.merge(test_weak, on='item_id')

features += ['mean','std','min','max','ratio'] + list(train_weak.columns)[1:]

booster = {}
booster['boosting_type'] = 'gbdt'
booster['objective'] = 'regression'
booster['learning_rate'] = 0.02
booster['num_leaves'] = 64
booster['max_depth'] = -1
booster['min_child_weight'] = 20
booster['max_bin'] = 512
booster['subsample'] = 0.7
booster['subsample_freq'] = 1
booster['colsample_bytree'] = 0.6
booster['min_split_gain'] = 0
booster['nthread'] = 15
booster['verbose'] = 0
booster['metric'] = 'root_mean_squared_error'

def importance(model):
    global features
    importance = model.feature_importance(importance_type='gain')
    importance = pd.DataFrame(importance, columns=['importance'])
    importance['feature'] = features
    importance['importance'] = importance['importance'] / importance['importance'].max()
    importance = importance[['feature', 'importance']]
    importance = importance.sort_values(by='importance', ascending=False)
    print(importance)
    return None

def execute(mode, logger = False):
    global actual, score, features, booster
    train_data = pd.read_csv('../data/data/files/train_{}.csv'.format(mode))
    valid_data = pd.read_csv('../data/data/files/valid_{}.csv'.format(mode))
    test_data = pd.read_csv('../data/data/files/score.csv')
    train_data = train_data.merge(actual, on='item_id')
    valid_data = valid_data.merge(actual, on='item_id')
    test_data = test_data.merge(score, on='item_id')
    train_matrix = lgb.Dataset(train_data[features], train_data['actual'])
    valid_matrix = lgb.Dataset(valid_data[features], valid_data['actual'])
    test_matrix = lgb.Dataset(test_data[features])
    params = {}
    params['params'] = booster
    params['train_set'] = train_matrix
    params['valid_sets'] = [train_matrix, valid_matrix]
    params['num_boost_round'] = 5000
    params['early_stopping_rounds'] = 50
    params['verbose_eval'] = 100
    model = lgb.train(**params)
    valid_data['score'] = model.predict(valid_data[features])
    test_data['score'] = model.predict(test_data[features])
    valid_data['score'] = valid_data['score'].clip(0,1)
    test_data['score'] = test_data['score'].clip(0,1)
    valid_data = valid_data[['item_id','actual','score']]
    test_data = test_data[['item_id','score']]
    print('dataset:', train_data.shape, valid_data.shape, test_data.shape)
    if logger:
        importance(model)
    return valid_data, test_data

train_1, test_1 = execute(1, True)
train_2, test_2 = execute(2)
train_3, test_3 = execute(3)
train_4, test_4 = execute(4)
train_5, test_5 = execute(5)

train_data = train_1.append(train_2).append(train_3).append(train_4).append(train_5)
train_data = train_data.sort_values(by='item_id').reset_index(drop=True)
rmse = np.sqrt(mean_squared_error(train_data['actual'], train_data['score']))
print('rmse:', round(rmse,5))
print('mean:', train_data.score.mean().round(5))
train_data = train_data[['item_id','score']].rename(columns={'score':'deal_probability'})

test_data = test_1.append(test_2).append(test_3).append(test_4).append(test_5)
test_data = test_data.groupby('item_id')['score'].mean().reset_index()
print('mean:', test_data.score.mean().round(5))
test_data = test_data[['item_id','score']].rename(columns={'score':'deal_probability'})

train_data.to_csv('../data/insample/ensemble.csv', index=False)
test_data.to_csv('../data/outsample/ensemble.csv', index=False)