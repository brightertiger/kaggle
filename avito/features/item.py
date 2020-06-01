import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

columns = ['item_id','item_seq_number', 'param_1', 'region','title','parent_category_name']
columns += ['param_2','param_3','city','category_name','price']
actuals = pd.read_csv('../data/download/train.csv', usecols=['item_id','deal_probability'])

train_data = pd.read_csv('../data/download/train.csv', usecols=columns)
test_data = pd.read_csv('../data/download/test.csv', usecols=columns)
source_1 = train_data.append(test_data)

train_other = pd.read_csv('../data/download/train_active.csv', usecols=columns)
test_other = pd.read_csv('../data/download/test_active.csv', usecols=columns)
source_2 = train_other.append(test_other)
data = source_1.append(source_2)

feature = data.groupby(['item_seq_number'])['item_id'].count().reset_index()
feature = feature.rename(columns={'item_id':'item_1'})
driver = source_1.merge(feature, on=['item_seq_number'], how='left')
driver = driver.fillna(0)
driver = driver[['item_id','item_1']]
driver.to_csv('../data/data/features/item/item_1.csv', index=False)

# In[35]:

feature = data.groupby(['parent_category_name','item_seq_number'])['item_id'].count().reset_index()
feature = feature.rename(columns={'item_id':'item_2'})
driver = source_1.merge(feature, on=['parent_category_name','item_seq_number'], how='left')
driver = driver.fillna(0)
driver = driver[['item_id','item_2']]
driver.to_csv('../data/data/features/item/item_2.csv', index=False)

feature = data.groupby(['item_seq_number'])['price'].mean().reset_index()
driver = source_1.merge(feature, on=['item_seq_number'], how='left')
driver['item_3'] = driver['price_x'] / driver['price_y'].clip_lower(1.)
driver = driver.fillna(0)
driver = driver[['item_id','item_3']]
driver.to_csv('../data/data/features/item/item_3.csv', index=False)