import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

actual = pd.read_csv('../data/download/train.csv', usecols=['item_id','deal_probability'])
actual = actual.rename(columns={'deal_probability':'actual'})
print('data:', actual.shape)

score = pd.read_csv('../data/download/test.csv', usecols=['item_id'])
print('data:', score.shape)

features = []

def merge(file, name):
    global actual, features
    path = '../data/insample/' + file
    file = pd.read_csv(path)
    file = file.rename(columns={'deal_probability':name})
    actual = actual.merge(file, on='item_id')
    features += [name]
    rmse = np.sqrt(mean_squared_error(actual['actual'], actual[name]))
    print('rmse:', round(rmse,5))
    return None

merge('scores/lightgbm_1.csv', 'model_1')
merge('scores/lightgbm_3.csv', 'model_2')
merge('team/moiz_2204.csv', 'model_3')
merge('team/moiz_2206.csv', 'model_4')
merge('team/moiz_2209.csv', 'model_5')
merge('team/moiz_2233.csv', 'model_6')
merge('team/svj_2179.csv', 'model_7')
merge('team/rk_2208.csv', 'model_8')

print('data:', actual.shape)
actual = actual.sort_values(by='item_id').reset_index(drop=True)
actual.head()

def merge(file, name):
    global score
    path = '../data/outsample/' + file
    file = pd.read_csv(path)
    file = file.rename(columns={'deal_probability':name})
    score = score.merge(file, on='item_id')
    return None

merge('scores/lightgbm_1.csv', 'model_1')
merge('scores/lightgbm_3.csv', 'model_2')
merge('team/moiz_2204.csv', 'model_3')
merge('team/moiz_2206.csv', 'model_4')
merge('team/moiz_2209.csv', 'model_5')
merge('team/moiz_2233.csv', 'model_6')
merge('team/svj_2179.csv', 'model_7')
merge('team/rk_2208.csv', 'model_8')

print('data:', score.shape)
score = score.sort_values(by='item_id').reset_index(drop=True)

actual[features].corr()
score[features].corr()

def execute(mode):
    global actual, score, features
    train_data = pd.read_csv('../data/data/files/train_{}.csv'.format(mode))
    valid_data = pd.read_csv('../data/data/files/valid_{}.csv'.format(mode))
    test_data = pd.read_csv('../data/data/files/score.csv')
    train_data = train_data.merge(actual, on='item_id')
    valid_data = valid_data.merge(actual, on='item_id')
    test_data = test_data.merge(score, on='item_id')
    model = LinearRegression(fit_intercept=True)
    model.fit(train_data[features], train_data['actual'])
    print('weights:', [round(x, 4) for x in  model.coef_])
    valid_data['score'] = model.predict(valid_data[features])
    test_data['score'] = model.predict(test_data[features])
    valid_data['score'] = valid_data['score'].clip(0,1)
    test_data['score'] = test_data['score'].clip(0,1)
    valid_data = valid_data[['item_id','actual','score']]
    test_data = test_data[['item_id','score']]
    print('dataset:', train_data.shape, valid_data.shape, test_data.shape)
    return valid_data, test_data

train_1, test_1 = execute(1)
train_2, test_2 = execute(2)
train_3, test_3 = execute(3)
train_4, test_4 = execute(4)
train_5, test_5 = execute(5)

train_data = train_1.append(train_2).append(train_3).append(train_4).append(train_5)
train_data = train_data.sort_values(by='item_id').reset_index(drop=True)
rmse = np.sqrt(mean_squared_error(train_data['actual'], train_data['score']))
print('rmse:', round(rmse,5))
print('mean:', train_data.score.mean().round(5))
train_data = train_data[['item_id','score']].rename(columns={'score':'deal_probability'})

test_data = test_1.append(test_2).append(test_3).append(test_4).append(test_5)
test_data = test_data.groupby('item_id')['score'].mean().reset_index()
print('mean:', test_data.score.mean().round(5))
test_data = test_data[['item_id','score']].rename(columns={'score':'deal_probability'})

train_data.to_csv('../data/insample/team/team_regress.csv', index=False)
test_data.to_csv('../data/outsample/team/team_regress.csv', index=False)