import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

columns = ['item_id','user_id','region','city','parent_category_name','category_name']
columns += ['param_1','param_2','param_3']
columns += ['price','item_seq_number','user_type','image_top_1']

train_data = pd.read_csv('../data/download/train.csv', usecols = columns + ['deal_probability'])
test_data = pd.read_csv('../data/download/test.csv', usecols = columns)
full_data = train_data.append(test_data)

train_data['missing'] = train_data.isnull().sum(axis=1)
test_data['missing'] = train_data.isnull().sum(axis=1)

train_data['image_top_1'] = train_data['image_top_1'].fillna(3067).astype('int')
test_data['image_top_1'] = test_data['image_top_1'].fillna(3067).astype('int')

user = pd.read_csv('../data/data/features/user_features.csv')
train_data = train_data.merge(user, on='user_id', how='left').drop('user_id', axis=1)
test_data = test_data.merge(user, on='user_id', how='left').drop('user_id', axis=1)

for feat in list(user.columns)[1:]:
    train_data[feat] = train_data[feat].fillna(-1)
    test_data[feat] = test_data[feat].fillna(-1)

def encode(variable):
    global train_data, test_data, full_data
    encode = LabelEncoder()
    encode.fit(full_data[variable].fillna('none'))
    train_data[variable] = encode.transform(train_data[variable].fillna('none'))
    test_data[variable] = encode.transform(test_data[variable].fillna('none'))
    train_data = train_data.drop(variable, axis=1)
    test_data = test_data.drop(variable, axis=1)
    return None

encode('region')
encode('city')
encode('parent_category_name')
encode('category_name')
encode('param_1')
encode('param_2')
encode('param_3')
encode('user_type')

print('train:', train_data.shape)
print('test:', test_data.shape)

def merge(path):
    global train_data, test_data
    path = '../data/data/features/' + path
    feature = pd.read_csv(path)
    train_data = train_data.merge(feature, on='item_id')
    test_data = test_data.merge(feature, on='item_id')
    return None

merge('relative/relative_1.csv')
merge('relative/relative_2.csv')
merge('relative/relative_3.csv')
merge('relative/relative_4.csv')
merge('relative/relative_5.csv')
merge('relative/relative_6.csv')
merge('count/count_1.csv')
merge('count/count_2.csv')
merge('count/count_3.csv')
merge('count/count_4.csv')
merge('count/count_5.csv')
merge('count/count_6.csv')
merge('count/count_7.csv')
merge('count/count_8.csv')
merge('nlp/title.csv')
merge('nlp/description.csv')
merge('duplicate/duplicate_1.csv')
merge('duplicate/duplicate_2.csv')
merge('duplicate/duplicate_3.csv')
merge('time/time.csv')
merge('time/renew_1.csv')
merge('time/renew_2.csv')
merge('ridge/ridge_1.csv')
merge('ridge/ridge_2.csv')
merge('ridge/svd.csv')
merge('user/user_ridge.csv')
merge('item/item_1.csv')
merge('item/item_2.csv')
merge('item/item_3.csv')
merge('encode/encode_1.csv')
merge('encode/encode_2.csv')
merge('encode/encode_3.csv')
merge('encode/encode_4.csv')
merge('encode/encode_5.csv')
merge('encode/encode_6.csv')
merge('encode/encode_7.csv')
merge('encode/encode_8.csv')
merge('encode/encode_9.csv')
merge('encode/encode_10.csv')
merge('encode/encode_11.csv')
merge('encode/encode_12.csv')
merge('encode/encode_13.csv')
merge('encode/encode_14.csv')
merge('encode/encode_15.csv')

print('train:', train_data.shape)
print('test:', test_data.shape)
train_data.to_csv('../data/data/lightgbm/train_data_2.csv', index=False)
test_data.to_csv('../data/data/lightgbm/test_data_2.csv', index=False)

for i in range(5):
    i += 1
    train_idx = pd.read_csv('../data/data/files/train_{}.csv'.format(i))
    valid_idx = pd.read_csv('../data/data/files/valid_{}.csv'.format(i))
    train_sub = train_idx.merge(train_data, on='item_id')
    valid_sub = valid_idx.merge(train_data, on='item_id')
    print('dataset:', train_sub.shape, valid_sub.shape)
    train_sub.to_csv('../data/data/lightgbm/dataset/train_2_{}.csv'.format(i), index=False)
    valid_sub.to_csv('../data/data/lightgbm/dataset/valid_2_{}.csv'.format(i), index=False)