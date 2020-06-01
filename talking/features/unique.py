import pandas as pd
import numpy as np

dtypes = {}
dtypes['ip'] = 'uint32'
dtypes['app'] = 'uint16'
dtypes['device'] = 'uint16'
dtypes['os'] = 'uint16'
dtypes['channel'] = 'uint16'
usecols = list(dtypes.keys()) + ['click_time']

train_data = pd.read_csv('../data/download/train.csv', dtype=dtypes, usecols=usecols)
test_data = pd.read_csv('../data/download/test_supplement.csv', dtype=dtypes, usecols=usecols)
data = train_data.append(test_data)
del train_data, test_data

data['click_time'] = pd.to_datetime(data['click_time'])
data['day'] = data['click_time'].dt.day
data['hour'] = data['click_time'].dt.hour
data = data[['day','hour','ip','app','os','device','channel']]
data.head()

def count(feature,count):
    global data
    grouped = data.groupby(feature)[count].nunique().reset_index()
    name = '_'.join(feature) + '_' + count +'_unq'
    grouped.columns = feature + [name]
    grouped.to_feather('../data/data/features/unique/' + name +'.feather')
    return None

count(['ip'], 'app')
count(['ip'], 'channel')