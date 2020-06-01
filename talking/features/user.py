import pandas as pd
import numpy as np

dtypes = {}
dtypes['ip'] = 'uint32'
dtypes['app'] = 'uint16'
dtypes['device'] = 'uint16'
dtypes['os'] = 'uint16'
dtypes['channel'] = 'uint16'
usecols = list(dtypes.keys()) 
train_data = pd.read_csv('../data/download/train.csv', dtype=dtypes, usecols=usecols)
train_data['click_id'] = train_data.index * -1
train_data = train_data[['click_id','ip','app','os','device','channel']]

dtypes = {}
dtypes['ip'] = 'uint32'
dtypes['app'] = 'uint16'
dtypes['device'] = 'uint16'
dtypes['os'] = 'uint16'
dtypes['channel'] = 'uint16'
dtypes['click_id'] = 'uint32'
usecols = list(dtypes.keys())
test_data = pd.read_csv('../data/download/test_supplement.csv', dtype=dtypes, usecols=usecols)
test_data = test_data[['click_id','ip','app','os','device','channel']]

mapping = pd.read_feather('../data/data/files/mapping.feather')
data = train_data.append(test_data)
del train_data, test_data

user_count = data.groupby(['ip','device','os'])['channel'].count().reset_index()
user_count.columns = ['ip','device','os','user_count']
user_count.to_feather('../data/data/features/count/user_count.feather')

user_count = data.groupby(['ip','device','os','app'])['channel'].count().reset_index()
user_count.columns = ['ip','device','os','app','user_app_count']
user_count.to_feather('../data/data/features/count/user_app_count.feather')