import pandas as pd
import numpy as np

def rank(features):
    global data
    name = '_'.join(features)
    grouped = data.groupby(features)['is_attributed'].sum().reset_index()
    grouped = grouped[grouped['is_attributed'] > 50]
    grouped[name + '_rnk'] = pd.qcut(grouped['is_attributed'],10, duplicates='drop', labels=False)
    grouped = grouped[features + [name + '_rnk']]
    grouped = grouped.reset_index(drop=True)
    grouped.to_feather('../data/data/features/rank/' + name + '.feather')
    return None

dtypes = {}
dtypes['ip'] = 'uint32'
dtypes['app'] = 'uint16'
dtypes['device'] = 'uint16'
dtypes['os'] = 'uint16'
dtypes['channel'] = 'uint16'
usecols = list(dtypes.keys()) + ['click_time', 'is_attributed']
data = pd.read_csv('../data/download/train.csv', dtype=dtypes, usecols=usecols)
data['click_id'] = data.index
data['click_time'] = pd.to_datetime(data['click_time'])
data['is_valid'] = False
data.loc[data['click_time'] > '2017-11-09 04:00:00','is_valid'] = True 
data = data.drop('click_time', axis=1)
data = data[data['is_valid'] == False]

rank(['ip'])
rank(['app','channel'])
rank(['app','os'])
rank(['channel', 'os'])