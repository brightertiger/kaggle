import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

columns = ['item_id','user_id', 'param_1', 'region','title','parent_category_name']
columns += ['param_2','param_3','city','category_name']
actuals = pd.read_csv('../data/download/train.csv', usecols=['item_id','deal_probability'])

train_data = pd.read_csv('../data/download/train.csv', usecols=columns)
test_data = pd.read_csv('../data/download/test.csv', usecols=columns)
source_1 = train_data.append(test_data)

train_other = pd.read_csv('../data/download/train_active.csv', usecols=columns)
test_other = pd.read_csv('../data/download/test_active.csv', usecols=columns)
source_2 = train_other.append(test_other)

data = source_1.append(source_2)

feature = data.groupby('user_id')['title'].nunique().reset_index()
feature = feature.rename(columns={'title':'user_1'})
feature = source_1[['item_id','user_id']].merge(feature, on='user_id',how='left').fillna(0)
feature = feature.drop('user_id', axis=1)
feature.to_csv('../data/data/features/user/user_1.csv', index=False)

feature = data.groupby('user_id')['category_name'].nunique().reset_index()
feature = feature.rename(columns={'category_name':'user_2'})
feature = source_1[['item_id','user_id']].merge(feature, on='user_id',how='left').fillna(0)
feature = feature.drop('user_id', axis=1)
feature.to_csv('../data/data/features/user/user_2.csv', index=False)

feature = data.groupby('user_id')['param_1'].nunique().reset_index()
feature = feature.rename(columns={'param_1':'user_3'})
feature = source_1[['item_id','user_id']].merge(feature, on='user_id',how='left').fillna(0)
feature = feature.drop('user_id', axis=1)
feature.to_csv('../data/data/features/user/user_3.csv', index=False)

feature = data.copy()
feature['null'] = feature.isnull().sum(axis=1)
feature = feature.groupby('user_id')['null'].sum().reset_index()
feature = feature.rename(columns={'null':'user_4'})
feature = source_1[['item_id','user_id']].merge(feature, on='user_id',how='left').fillna(0)
feature = feature.drop('user_id', axis=1)
feature.to_csv('../data/data/features/user/user_4.csv', index=False)

def normalize(feature):
    global matrix
    upper = matrix[feature].quantile(0.70)
    matrix[feature] = matrix[feature].clip_upper(upper)
    mean = matrix[feature].mean()
    sdev = matrix[feature].std()
    matrix[feature] = (matrix[feature] - mean) / sdev
    return matrix

user_1 = pd.read_csv('../data/data/features/user/user_1.csv')
user_2 = pd.read_csv('../data/data/features/user/user_2.csv')
user_3 = pd.read_csv('../data/data/features/user/user_3.csv')
user_4 = pd.read_csv('../data/data/features/user/user_4.csv')

matrix = user_1.merge(user_2, on='item_id')
matrix = matrix.merge(user_3, on='item_id')
matrix = matrix.merge(user_4, on='item_id')

matrix = normalize('user_1')
matrix = normalize('user_2')
matrix = normalize('user_3')
matrix = normalize('user_4')

params = {}
params['alpha'] = 0.00000001
params['fit_intercept'] = True
params['normalize'] = False
params['copy_X'] = True
params['max_iter'] = None
params['tol'] = 0.0001
params['solver'] = 'auto'
params['random_state'] = 2017

def execute(mode):
    global vectorizer, train_data, test_data, params
    train_idx = pd.read_csv('../data/data/files/train_{}.csv'.format(mode))
    valid_idx = pd.read_csv('../data/data/files/valid_{}.csv'.format(mode))
    test_idx = pd.read_csv('../data/download/test.csv', usecols=['item_id'])
    train_subset = matrix.merge(train_idx, on='item_id')
    valid_subset = matrix.merge(valid_idx, on='item_id')
    test_subset = matrix.merge(test_idx, on='item_id')
    labels = actuals.merge(train_idx, on='item_id')
    valid_driver = valid_subset[['item_id']].copy()
    test_driver =  test_idx[['item_id']].copy()
    train_matrix = train_subset.iloc[:,1:]
    valid_matrix = valid_subset.iloc[:,1:]
    test_matrix = test_subset.iloc[:,1:]
    model = Ridge(**params)
    model.fit(train_matrix, labels['deal_probability'])
    valid_driver['ridge_score'] = model.predict(valid_matrix)
    test_driver['ridge_score'] = model.predict(test_matrix)
    return valid_driver, test_driver

valid_1, test_1 = execute(1)
valid_2, test_2 = execute(2)
valid_3, test_3 = execute(3)
valid_4, test_4 = execute(4)
valid_5, test_5 = execute(5)

valid_scores = valid_1.append(valid_2).append(valid_3).append(valid_4).append(valid_5)
perform = actuals.merge(valid_scores, on='item_id')
np.sqrt(mean_squared_error(perform['deal_probability'], perform['ridge_score']))
test_scores = test_1.append(test_2).append(test_3).append(test_4).append(test_5)
test_scores = test_scores.groupby('item_id')['ridge_score'].mean().reset_index()

full_scores = valid_scores.append(test_scores)
full_scores = full_scores.rename({'ridge_score':'user_model'}, axis=1)
print('scores:', valid_scores.shape, test_scores.shape, full_scores.shape)
full_scores.to_csv('../data/data/features/user/user_ridge.csv', index=False)