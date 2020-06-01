import pandas as pd
import numpy as np

def next_click(data):
    data['click_time'] = data['click_time'].astype(np.int64) // 10 ** 9
    data = data.sort_values(by=['ip','app','device','os','click_time'])
    data['next_time'] = data.groupby(['ip','app','device','os'])['click_time'].shift(-1)
    data['next_click'] = data['next_time'] - data['click_time']
    data['next_click'] = data['next_click'].fillna(-1.)
    return data[['click_id','next_click']]

dtypes = {}
dtypes['ip'] = 'uint32'
dtypes['app'] = 'uint16'
dtypes['device'] = 'uint16'
dtypes['os'] = 'uint16'
usecols = list(dtypes.keys()) + ['click_time'] 
train_data = pd.read_csv('../data/download/train.csv', dtype=dtypes, usecols=usecols)
train_data['click_time'] = pd.to_datetime(train_data['click_time'])
train_data['click_id'] = train_data.index * -1
train_data = train_data[['click_id','ip','app','os','device','click_time']]

dtypes = {}
dtypes['ip'] = 'uint32'
dtypes['app'] = 'uint16'
dtypes['device'] = 'uint16'
dtypes['os'] = 'uint16'
usecols = list(dtypes.keys()) + ['click_time']
test_data = pd.read_csv('../data/download/test_supplement.csv', dtype=dtypes, usecols=usecols)
test_data['click_time'] = pd.to_datetime(test_data['click_time'])
test_data['click_id'] = test_data.index
test_data = test_data[['click_id','ip','app','os','device','click_time']]

mapping = pd.read_feather('../data/data/files/mapping.feather')
data = train_data.append(test_data)
del train_data, test_data
feature = next_click(data)

feature_train = feature[feature['click_id'] < 0].copy()
feature_train['click_id'] = feature_train['click_id'] * -1
feature_train['click_id'] = feature_train['click_id'].astype('uint32')
feature_train = feature_train.reset_index(drop=True)
feature_train = feature_train[['click_id', 'next_click']]
feature_train.to_feather('../data/data/features/next_click/next_click_train.feather')

feature_test = feature[feature['click_id'] > 0].iloc[1:,:].copy()
feature_test = feature_test.rename(columns={'click_id':'old_id'})
feature_test = feature_test.merge(mapping, on='old_id').drop('old_id', axis=1)
feature_test = feature_test[['click_id', 'next_click']]
feature_test = feature_test.reset_index(drop=True)
feature_test.to_feather('../data/data/features/next_click/next_click_test.feather')

def next_click(data):
    data['click_time'] = data['click_time'].astype(np.int64) // 10 ** 9
    data = data.sort_values(by=['ip','app','click_time'])
    data['next_time'] = data.groupby(['ip','app'])['click_time'].shift(-1)
    data['next_click'] = data['next_time'] - data['click_time']
    data['next_click'] = data['next_click'].fillna(-1.)
    return data[['click_id','next_click']]

dtypes = {}
dtypes['ip'] = 'uint32'
dtypes['app'] = 'uint16'
usecols = list(dtypes.keys()) + ['click_time'] 
train_data = pd.read_csv('../data/download/train.csv', dtype=dtypes, usecols=usecols)
train_data['click_time'] = pd.to_datetime(train_data['click_time'])
train_data['click_id'] = train_data.index * -1
train_data = train_data[['click_id','ip','app','click_time']]

dtypes = {}
dtypes['ip'] = 'uint32'
dtypes['app'] = 'uint16'
usecols = list(dtypes.keys()) + ['click_time']
test_data = pd.read_csv('../data/download/test_supplement.csv', dtype=dtypes, usecols=usecols)
test_data['click_time'] = pd.to_datetime(test_data['click_time'])
test_data['click_id'] = test_data.index
test_data = test_data[['click_id','ip','app','click_time']]

mapping = pd.read_feather('../data/data/files/mapping.feather')
data = train_data.append(test_data)
del train_data, test_data
feature = next_click(data)
feature = feature.rename(columns={'next_click':'next_click_1'})

feature_train = feature[feature['click_id'] < 0].copy()
feature_train['click_id'] = feature_train['click_id'] * -1
feature_train['click_id'] = feature_train['click_id'].astype('uint32')
feature_train = feature_train.reset_index(drop=True)
feature_train = feature_train[['click_id', 'next_click_1']]
feature_train.to_feather('../data/data/features/next_click/next_click_train_1.feather')

feature_test = feature[feature['click_id'] > 0].iloc[1:,:].copy()
feature_test = feature_test.rename(columns={'click_id':'old_id'})
feature_test = feature_test.merge(mapping, on='old_id').drop('old_id', axis=1)
feature_test = feature_test[['click_id', 'next_click_1']]
feature_test = feature_test.reset_index(drop=True)
feature_test.to_feather('../data/data/features/next_click/next_click_test_1.feather')