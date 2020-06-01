import pandas as pd
import numpy as np

columns = ['parent_category_name', 'category_name', 'price', 'user_type','region','city','param_1']
added = ['item_id']

actual = pd.read_csv('../data/download/train.csv', usecols= ['item_id','deal_probability'])

train_data = pd.read_csv('../data/download/train.csv', usecols= added + columns)
test_data = pd.read_csv('../data/download/test.csv', usecols= added + columns)
full_data = train_data.append(test_data)
del train_data, test_data 

train_other = pd.read_csv('../data/download/train_active.csv', usecols=columns)
test_other = pd.read_csv('../data/download/test_active.csv', usecols=columns)
other_data = train_other.append(test_other)
del train_other, test_other

feature = other_data.groupby(['parent_category_name'])['price'].mean().reset_index()
driver = full_data[['item_id','parent_category_name','price']].copy()
driver = driver.merge(feature, on=['parent_category_name'], how='left')
driver['relative_1'] = driver['price_x'] / driver['price_y'].clip_lower(1.)
driver = driver[['item_id','relative_1']]
driver.to_csv('../data/data/features/relative/relative_1.csv', index=False)

feature = other_data.groupby(['category_name'])['price'].mean().reset_index()
driver = full_data[['item_id','category_name','price']].copy()
driver = driver.merge(feature, on=['category_name'], how='left')
driver['relative_2'] = driver['price_x'] / driver['price_y'].clip_lower(1.)
driver = driver[['item_id','relative_2']]
driver.to_csv('../data/data/features/relative/relative_2.csv', index=False)

feature = other_data.groupby(['region'])['price'].mean().reset_index()
driver = full_data[['item_id','region','price']].copy()
driver = driver.merge(feature, on=['region'], how='left')
driver['relative_3'] = driver['price_x'] / driver['price_y'].clip_lower(1.)
driver = driver[['item_id','relative_3']]
driver.to_csv('../data/data/features/relative/relative_3.csv', index=False)

feature = other_data.groupby(['city'])['price'].mean().reset_index()
driver = full_data[['item_id','city','price']].copy()
driver = driver.merge(feature, on=['city'], how='left')
driver['relative_4'] = driver['price_x'] / driver['price_y'].clip_lower(1.)
driver = driver[['item_id','relative_4']]
driver.to_csv('../data/data/features/relative/relative_4.csv', index=False)

feature = other_data.groupby(['param_1'])['price'].mean().reset_index()
driver = full_data[['item_id','param_1','price']].copy()
driver = driver.merge(feature, on=['param_1'], how='left')
driver['relative_6'] = driver['price_x'] / driver['price_y'].clip_lower(1.)
driver = driver[['item_id','relative_6']]
driver.to_csv('../data/data/features/relative/relative_6.csv', index=False)

feature = other_data.groupby(['city','category_name'])['price'].mean().reset_index()
driver = full_data[['item_id','city','category_name','price']].copy()
driver = driver.merge(feature, on=['city','category_name'], how='left')
driver['relative_5'] = driver['price_x'] / driver['price_y'].clip_lower(1.)
driver = driver[['item_id','relative_5']]
driver.to_csv('../data/data/features/relative/relative_5.csv', index=False)