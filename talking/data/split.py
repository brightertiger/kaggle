import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

dtypes = {}
dtypes['ip'] = 'uint32'
dtypes['app'] = 'uint16'
dtypes['device'] = 'uint16'
dtypes['os'] = 'uint16'
dtypes['channel'] = 'uint16'
dtypes['is_attributed'] = 'uint8'
dtypes['click_id'] = 'uint32'
usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
train_data = pd.read_csv('../data/download/train.csv', dtype=dtypes, usecols=usecols)
train_data['click_time'] = pd.to_datetime(train_data['click_time'])
train_data['hour'] = train_data['click_time'].map(lambda x : x.hour)
train_data['keep_hour'] = train_data['hour'].isin([4,5,9,10,13,14])
train_data['keep_row'] = train_data['click_time'] > '2017-11-08 12:00:00'
train_data['keep'] = train_data['keep_row'] + train_data['keep_hour']
train_data = train_data[train_data['keep'] > 0]
train_data['click_id'] = train_data.index

dtypes = {}
dtypes['ip'] = 'uint32'
dtypes['app'] = 'uint16'
dtypes['device'] = 'uint16'
dtypes['os'] = 'uint16'
dtypes['channel'] = 'uint16'
dtypes['click_id'] = 'uint32'
usecols=['click_id','ip','app','device','os','channel','click_time']
test_data = pd.read_csv('../data/download/test.csv', dtype=dtypes, usecols=usecols)
test_data['click_time'] = pd.to_datetime(test_data['click_time'])

test_hour = test_data['click_time'].map(lambda x : x.hour)
hour = test_hour.value_counts().reset_index()
plt.figure(figsize=(10,5))
sns.set_style("whitegrid")
sns.barplot(x='index', y='click_time', data=hour)
plt.show()

train_data['valid_date'] = train_data['click_time'] > '2017-11-09'
train_data['is_valid'] = train_data['valid_date'] * train_data['keep_hour']
valid_data = train_data[train_data['is_valid'] == True]
train_data = train_data[train_data['is_valid'] == False]
train_data = train_data.drop(['valid_date','is_valid'], axis=1)
valid_data = valid_data.drop(['valid_date','is_valid'], axis=1)
train_data = train_data.drop(['keep_row','keep','keep_hour'], axis=1)
valid_data = valid_data.drop(['keep_row','keep','keep_hour'], axis=1)
print('train data:', train_data.shape, train_data['is_attributed'].mean())
print('valid data:', valid_data.shape, valid_data['is_attributed'].mean())

valid_hour = valid_data['click_time'].map(lambda x : x.hour)
hour = valid_hour.value_counts().reset_index()
plt.figure(figsize=(10,5))
sns.set_style("whitegrid")
sns.barplot(x='index', y='click_time', data=hour)
plt.show()

train_data['day'] = train_data['click_time'].dt.day
valid_data['day'] = valid_data['click_time'].dt.day
train_data = train_data[['click_id','is_attributed','day','hour','ip','app','os','device','channel']]
valid_data = valid_data[['click_id','is_attributed','day','hour','ip','app','os','device','channel']]

train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
train_data.to_feather('../data/data/files/train_data.feather')
valid_data.to_feather('../data/data/files/valid_data.feather')

test_data['day'] = test_data['click_time'].dt.day
test_data['hour'] = test_data['click_time'].dt.hour
test_data = test_data[['click_id','day','hour','ip','app','os','device','channel']]
test_data.to_feather('../data/data/files/score_data.feather')