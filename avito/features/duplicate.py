import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

usecols = ['item_id','user_id','title','description','activation_date']

train_data = pd.read_csv('../data/download/train.csv', usecols=usecols)
train_active = pd.read_csv('../data/download/train_active.csv', usecols=usecols)
test_data = pd.read_csv('../data/download/test.csv', usecols=usecols)
test_active = pd.read_csv('../data/download/test_active.csv', usecols=usecols)
train_actuals = pd.read_csv('../data/download/train.csv', usecols=['item_id','deal_probability'])

encode_1 = LabelEncoder()
encode_2 = LabelEncoder()

data = train_data.append(train_active).append(test_data).append(test_active)
data['title'] = encode_1.fit_transform(data['title'].fillna(' '))
data['description'] = encode_2.fit_transform(data['description'].fillna(' '))

data['activation_date'] = pd.to_datetime(data['activation_date'])
data['activation_date'] = data['activation_date'] - data['activation_date'].min()
data['activation_date'] = data['activation_date'].dt.days
data.to_csv('../data/data/features/duplicate/raw.csv', index=False)

actuals = pd.read_csv('../data/download/train.csv', usecols=['item_id','deal_probability'])
usecols = ['item_id','user_id','title','activation_date']
data = pd.read_csv('../data/data/features/duplicate/raw.csv', usecols=usecols)

data['duplicate'] = data[['user_id','title']].duplicated()
data['duplicate'] = data['duplicate'].astype('int')
data = data.rename(columns={'duplicate':'duplicate_1'})
data = data[['item_id','duplicate_1']]
data['duplicate_1'].value_counts()
data.to_csv('../data/data/features/duplicate/duplicate_1.csv', index=False)

actuals = pd.read_csv('../data/download/train.csv', usecols=['item_id','deal_probability'])
usecols = ['item_id','title','activation_date']
data = pd.read_csv('../data/data/features/duplicate/raw.csv', usecols=usecols)

data['duplicate'] = data[['title']].duplicated()
data['duplicate'] = data['duplicate'].astype('int')
data = data.rename(columns={'duplicate':'duplicate_2'})
data = data[['item_id','duplicate_2']]
data['duplicate_2'].value_counts()
data.to_csv('../data/data/features/duplicate/duplicate_2.csv', index=False)

actuals = pd.read_csv('../data/download/train.csv', usecols=['item_id','deal_probability'])
usecols = ['item_id','title','description','activation_date']
data = pd.read_csv('../data/data/features/duplicate/raw.csv', usecols=usecols)

data['duplicate'] = data[['title','description']].duplicated()
data['duplicate'] = data['duplicate'].astype('int')
data = data.rename(columns={'duplicate':'duplicate_3'})
data = data[['item_id','duplicate_3']]
data['duplicate_3'].value_counts()
data.to_csv('../data/data/features/duplicate/duplicate_3.csv', index=False)