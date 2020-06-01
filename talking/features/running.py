import pandas as pd
import numpy as np
pd.options.mode.chained_assignment=None

dtypes = {}
dtypes['ip'] = 'uint32'
dtypes['app'] = 'uint16'
usecols = list(dtypes.keys()) + ['click_time','is_attributed']
data = pd.read_csv('../data/download/train.csv', dtype=dtypes, usecols=usecols)
data['click_id'] = data.index
data['click_time'] = pd.to_datetime(data['click_time'])
data = data[['click_id','click_time','ip','app','is_attributed']]

data['is_valid'] = False
data.loc[data['click_time'] > '2017-11-09 04:00:00','is_valid'] = True 
data = data.drop('click_time', axis=1)

def download_rate(features):
    global data
    name = '_'.join(features)
    print('data:', data.shape)
    train_data = data.copy()
    train_data[name + '_run'] = data.groupby(features)['is_attributed'].cumsum()
    train_data[name + '_run'] = train_data[name + '_run'] - train_data['is_attributed']
    train_data = train_data[['click_id', name + '_run']]
    train_data = train_data.reset_index(drop=True)
    print('train data done..')
    train_data.to_feather('../data/data/features/running/train_' + name + '.feather')
    del train_data
    valid_data = data[data['is_valid'] == False]
    valid_data = valid_data.groupby(features)['is_attributed'].sum().reset_index()
    valid_data.columns = features + [name + '_run']
    valid_data = valid_data.reset_index(drop=True)
    print('valid data done..')
    valid_data.to_feather('../data/data/features/running/valid_' + name + '.feather')
    del valid_data
    test_data = data.groupby(features)['is_attributed'].sum().reset_index()
    test_data.columns = features + [name + '_enc']
    test_data = test_data.reset_index(drop=True)
    print('test data done..')
    test_data.to_feather('../data/data/features/running/test_' + name + '.feather')
    del test_data
    return None

download_rate(['ip'])
download_rate(['ip','app'])