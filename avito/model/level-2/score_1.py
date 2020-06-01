import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

def execute(model, driver, data):
    driver['deal_probability'] = model.predict(data)
    driver['deal_probability'] = driver['deal_probability'].clip(0.,1.)
    return driver

def score(mode):
    model = lgb.Booster(model_file='../data/data/lightgbm/model_1/lightgbm_{}.model'.format(mode))
    valid_data = pd.read_csv('../data/data/lightgbm/dataset/valid_1_{}.csv'.format(mode))
    valid_driver = valid_data[['item_id']].copy()
    valid_data = valid_data.drop(['item_id','deal_probability'], axis=1)
    test_data = pd.read_csv('../data/data/lightgbm/test_data_1.csv'.format(mode))
    test_driver = test_data[['item_id']].copy()
    test_data = test_data.drop(['item_id'], axis=1)
    valid_score = execute(model, valid_driver, valid_data)
    test_score = execute(model, test_driver, test_data)
    valid_score.to_csv('../data/data/lightgbm/score_1/valid_{}.csv'.format(mode), index=False)
    test_score.to_csv('../data/data/lightgbm/score_1/test_{}.csv'.format(mode), index=False)
    return None


score(1)
score(2)
score(3)
score(4)
score(5)

dataset = pd.DataFrame([])
for i in range(5):
    i += 1
    temp = pd.read_csv('../data/data/lightgbm/score_1/valid_{}.csv'.format(i))
    dataset = dataset.append(temp)

dataset.to_csv('../data/insample/scores/lightgbm_1.csv', index=False)

actuals = pd.read_csv('../data/download/train.csv', usecols=['item_id','deal_probability'])
actuals = actuals.merge(dataset, on='item_id')
np.sqrt(mean_squared_error(actuals['deal_probability_x'], actuals['deal_probability_y']))

dataset = pd.DataFrame([])
for i in range(5):
    i += 1
    temp = pd.read_csv('../data/data/lightgbm/score_1/test_{}.csv'.format(i))
    dataset = dataset.append(temp)

dataset = dataset.groupby(['item_id'])['deal_probability'].mean().reset_index()
dataset.to_csv('../data/outsample/scores/lightgbm_1.csv', index=False)