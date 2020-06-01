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

def count(feature,count):
    global data
    grouped = data.groupby(feature)[count].count().reset_index()
    grouped.columns = feature + ['_'.join(feature) + '_cnt']
    for column in list(grouped.columns):
        if grouped[column].max() <= 65000:
            grouped[column] = grouped[column].astype('uint16')
    grouped.to_feather('../data/data/features/count/' + '_'.join(feature) +'.feather')
    return None

count(['ip'],'hour')
count(['app'],'hour')
count(['os'],'hour')
count(['ip','day','hour'], 'device')
count(['ip','app'], 'device')
count(['ip','app','os'],'device')
count(['ip','device'],'os')
count(['app','channel'],'os')
count(['ip','hour','os'],'channel')
count(['ip','hour','app'],'channel')