import pandas as pd
import numpy as np

columns = ['parent_category_name', 'category_name', 'user_type','region','city','user_id']
columns += ['param_1','param_2','param_3']
added = ['item_id']

train_data = pd.read_csv('../data/download/train.csv', usecols= added + columns + ['image_top_1'])
test_data = pd.read_csv('../data/download/test.csv', usecols= added + columns + ['image_top_1'])
full_data = train_data.append(test_data)
del train_data, test_data 

train_other = pd.read_csv('../data/download/train_active.csv', usecols=columns)
test_other = pd.read_csv('../data/download/test_active.csv', usecols=columns)
other_data = train_other.append(test_other)
del train_other, test_other

feature = other_data.groupby(['category_name','city'])['user_type'].count().reset_index()
driver = full_data[['item_id','category_name','city']].copy()
driver = driver.merge(feature, on=['category_name','city'], how='left')
driver = driver.rename(columns={'user_type':'count_1'})
driver = driver[['item_id','count_1']]
driver.to_csv('../data/data/features/count/count_1.csv', index=False)

feature = other_data.groupby(['category_name'])['user_type'].count().reset_index()
driver = full_data[['item_id','category_name']].copy()
driver = driver.merge(feature, on=['category_name'], how='left')
driver = driver.rename(columns={'user_type':'count_2'})
driver = driver[['item_id','count_2']]
driver.to_csv('../data/data/features/count/count_2.csv', index=False)

feature = other_data.groupby(['user_id'])['user_type'].count().reset_index()
driver = full_data[['item_id','user_id']].copy()
driver = driver.merge(feature, on=['user_id'], how='left')
driver = driver.rename(columns={'user_type':'count_3'})
driver = driver[['item_id','count_3']]
driver.to_csv('../data/data/features/count/count_3.csv', index=False)

feature = other_data.groupby(['user_id','category_name'])['user_type'].count().reset_index()
driver = full_data[['item_id','user_id','category_name']].copy()
driver = driver.merge(feature, on=['user_id','category_name'], how='left')
driver = driver.rename(columns={'user_type':'count_4'})
driver = driver[['item_id','count_4']]
driver.to_csv('../data/data/features/count/count_4.csv', index=False)

feature = full_data.groupby('image_top_1')['user_type'].count().reset_index()
driver = full_data[['item_id','image_top_1']].copy()
driver = driver.merge(feature, on=['image_top_1'], how='left')
driver = driver.rename(columns={'user_type':'count_5'})
driver = driver[['item_id','count_5']]
driver.to_csv('../data/data/features/count/count_5.csv', index=False)

feature = other_data.groupby('param_1')['user_type'].count().reset_index()
driver = full_data[['item_id','param_1']].copy()
driver = driver.merge(feature, on=['param_1'], how='left')
driver = driver.rename(columns={'user_type':'count_6'})
driver = driver[['item_id','count_6']]
driver.to_csv('../data/data/features/count/count_6.csv', index=False)

feature = other_data.groupby(['param_1','param_2'])['user_type'].count().reset_index()
driver = full_data[['item_id','param_1','param_2']].copy()
driver = driver.merge(feature, on=['param_1','param_2'], how='left')
driver = driver.rename(columns={'user_type':'count_7'})
driver = driver[['item_id','count_7']]
driver.to_csv('../data/data/features/count/count_7.csv', index=False)

feature = other_data.groupby(['param_1','param_2','param_3'])['user_type'].count().reset_index()
driver = full_data[['item_id','param_1','param_2','param_3']].copy()
driver = driver.merge(feature, on=['param_1','param_2','param_3'], how='left')
driver = driver.rename(columns={'user_type':'count_8'})
driver = driver[['item_id','count_8']]
driver.to_csv('../data/data/features/count/count_8.csv', index=False)