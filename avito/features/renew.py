import pandas as pd
import numpy as np

period_train = pd.read_csv('../data/download/periods_train.csv')
period_test = pd.read_csv('../data/download/periods_test.csv')
period = period_train.append(period_test).reset_index(drop=True)
period['date_from'] = pd.to_datetime(period['date_from'], errors='coerce')
period['date_to'] = pd.to_datetime(period['date_to'], errors='coerce')
period['delta'] = period['date_to'] - period['date_from']
period['delta'] = period['delta'].dt.days
period = period[['item_id','delta']]
period.to_csv('../data/data/features/time/raw.csv', index=False)
period = pd.read_csv('../data/data/features/time/raw.csv')

added = ['item_id']
columns = ['user_id','title']

train_data = pd.read_csv('../data/download/train.csv', usecols= added + columns)
test_data = pd.read_csv('../data/download/test.csv', usecols= added + columns)
full_data = train_data.append(test_data)
del train_data, test_data 

train_other = pd.read_csv('../data/download/train_active.csv', usecols=added + columns)
test_other = pd.read_csv('../data/download/test_active.csv', usecols=added + columns)
other_data = train_other.append(test_other)
del train_other, test_other

other_data = other_data.merge(period, on='item_id')

feature = other_data.groupby(['user_id'])['item_id'].count().reset_index()
feature = feature.rename(columns={'item_id':'renewal_1'})
feature = full_data.merge(feature, on='user_id', how='left')
feature = feature[['item_id','renewal_1']]
feature.to_csv('../data/data/features/time/renew_1.csv', index=False)

feature = other_data.groupby(['user_id','title'])['item_id'].count().reset_index()
feature = feature.rename(columns={'item_id':'renewal_2'})
feature = full_data.merge(feature, on=['user_id','title'], how='left')
feature = feature[['item_id','renewal_2']]
feature.to_csv('../data/data/features/time/renew_2.csv', index=False)