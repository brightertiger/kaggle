import pandas as pd
import numpy as np

columns = ['activation_date']
added = ['item_id']

train_data = pd.read_csv('../data/download/train.csv', usecols= added + columns)
test_data = pd.read_csv('../data/download/test.csv', usecols= added + columns)
full_data = train_data.append(test_data)
del train_data, test_data 

full_data['activation_date'] = pd.to_datetime(full_data['activation_date'])
full_data["week_day"] = full_data['activation_date'].dt.weekday
full_data = full_data.drop(['activation_date'], axis=1)
full_data.to_csv('../data/data/features/time/time.csv', index=False)