import pandas as pd
import numpy as np

dtypes = {}
dtypes['ip'] = 'uint32'
dtypes['app'] = 'uint16'
dtypes['device'] = 'uint16'
dtypes['os'] = 'uint16'
dtypes['channel'] = 'uint16'
dtypes['click_id'] = 'uint32'
usecols=['click_id','ip','app','device','os','channel','click_time']

old_data = pd.read_csv('../data/download/test_supplement.csv', usecols=usecols, dtype=dtypes)
new_data = pd.read_csv('../data/download/test.csv', usecols=usecols, dtype=dtypes)
old_data = old_data.rename(columns={'click_id':'old_id'})
new_data = new_data.rename(columns={'click_id':'new_id'})
id_map = old_data.merge(new_data, on=usecols[1:])
id_map = id_map[['old_id','new_id']]
id_map = id_map.rename(columns={'new_id':'click_id'})
id_map = id_map.groupby('click_id')['old_id'].max().reset_index()
id_map.to_feather('../data/data/files/mapping.feather')