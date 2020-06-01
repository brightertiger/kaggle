import pandas as pd
import numpy as np

columns = ['parent_category_name', 'category_name', 'region','city']
columns += ['param_1','param_2','param_3','image_top_1']

train_data = pd.read_csv('../data/download/train.csv', usecols = ['item_id'] + columns)
test_data = pd.read_csv('../data/download/test.csv', usecols = ['item_id'] + columns)
data = train_data.append(test_data)
print('data:', data.shape)

train_score = pd.read_csv('../data/score/catboost.csv')
test_score = pd.read_csv('../data/submit/catboost.csv')
score = train_score.append(test_score)
print('score:', score.shape)

data = data.merge(score, on='item_id')
print('data:', data.shape)

def encode(data, name, output):
    driver = data[['item_id'] + name].copy()
    print(driver.shape)
    feature = data[name + ['deal_probability']].copy()
    feature[name] = feature[name].astype(str).fillna('none')
    driver[name] = driver[name].astype(str).fillna('none')
    feature = feature.groupby(name)['deal_probability'].mean().reset_index()
    feature = feature.rename(columns={'deal_probability':output})
    driver = driver.merge(feature, on=name)
    driver = driver[['item_id', output]]
    print(driver.shape)
    driver.to_csv('../data/data/features/encode/' + output + '.csv', index=False)
    return None

encode(data, ['parent_category_name'],'encode_1')
encode(data, ['category_name'],'encode_2')
encode(data, ['region'],'encode_3')
encode(data, ['city'],'encode_4')
encode(data, ['param_1'],'encode_5')
encode(data, ['param_2'],'encode_6')
encode(data, ['param_3'],'encode_7')
encode(data, ['image_top_1'],'encode_8')
encode(data, ['param_1','param_2'],'encode_9')
encode(data, ['param_2','param_3'],'encode_10')
encode(data, ['param_1','param_2','param_3'],'encode_11')
encode(data, ['city','category_name'],'encode_12')
encode(data, ['city','param_1'],'encode_13')
encode(data, ['image_top_1','category_name'],'encode_14')
encode(data, ['image_top_1','param_1'],'encode_15')