import pandas as pd
import numpy as np
import re
import string
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import mean_squared_error

def concatenate(row):
    feature = [str(row['param_1']), str(row['param_2']), str(row['param_3'])]
    feature += [str(row['image_top_1']), str(row['city']), str(row['category_name'])]
    return ' '.join(feature)

columns = ['item_id', 'param_1','param_2','param_3','image_top_1','city','category_name']
train_data = pd.read_csv('../data/download/train.csv', usecols=columns + ['deal_probability'])
test_data = pd.read_csv('../data/download/test.csv', usecols=columns)

train_data['image_top_1'] = train_data['image_top_1'].fillna(-1).apply(lambda x: 'imt_' + str(x))
train_data['param_1'] = train_data['param_1'].fillna('none').apply(lambda x: 'p1_' + x)
train_data['param_2'] = train_data['param_2'].fillna('none').apply(lambda x: 'p2_' + x)
train_data['param_3'] = train_data['param_3'].fillna('none').apply(lambda x: 'p3_' + x)
train_data['city'] = train_data['city'].fillna('none').apply(lambda x: 'cty_' + x)
train_data['category_name'] = train_data['category_name'].fillna('none').apply(lambda x: 'cat_' + x)
train_data['text'] = train_data.apply(lambda row: concatenate(row),axis=1)

test_data['image_top_1'] = test_data['image_top_1'].fillna(-1).apply(lambda x: 'imt_' + str(x))
test_data['param_1'] = test_data['param_1'].fillna('none').apply(lambda x: 'p1_' + x)
test_data['param_2'] = test_data['param_2'].fillna('none').apply(lambda x: 'p2_' + x)
test_data['param_3'] = test_data['param_3'].fillna('none').apply(lambda x: 'p3_' + x)
test_data['city'] = test_data['city'].fillna('none').apply(lambda x: 'cty_' + x)
test_data['category_name'] = test_data['category_name'].fillna('none').apply(lambda x: 'cat_' + x)
test_data['text'] = test_data.apply(lambda row: concatenate(row),axis=1)

full_data = train_data[['text']].append(test_data[['text']])
full_data['text'] = full_data['text'].fillna('none')

def fetch(column):
    return lambda x: x[column]

params = {}
params['analyzer'] = 'word'
params['sublinear_tf'] = True
params['dtype'] = np.float32
params['norm'] = 'l2'
params['smooth_idf'] = False
params['ngram_range'] = (1,3)

pipe = []
pipe += [('set_1', TfidfVectorizer(**params, preprocessor=fetch('text'), max_features=50000))]
pipe += [('set_2', CountVectorizer(ngram_range=(1, 3), preprocessor=fetch('text'), max_features=50000))]

vectorizer = FeatureUnion(pipe)
vectorizer.fit(full_data.to_dict('records'))

params = {}
params['alpha'] = 40.
params['fit_intercept'] = True
params['normalize'] = False
params['copy_X'] = True
params['max_iter'] = None
params['tol'] = 0.001
params['solver'] = 'auto'
params['random_state'] = 2017

def execute(mode):
    global vectorizer, train_data, test_data, params
    train_idx = pd.read_csv('../data/data/files/train_{}.csv'.format(mode))
    valid_idx = pd.read_csv('../data/data/files/valid_{}.csv'.format(mode))
    train_subset = train_data.merge(train_idx, on='item_id')
    valid_subset = train_data.merge(valid_idx, on='item_id')
    valid_driver = valid_subset[['item_id']].copy()
    test_driver = test_data[['item_id']].copy()
    train_matrix = vectorizer.transform(train_subset.to_dict('records'))
    valid_matrix = vectorizer.transform(valid_subset.to_dict('records'))
    test_matrix = vectorizer.transform(test_data.to_dict('records'))
    model = Ridge(**params)
    model.fit(train_matrix, train_subset['deal_probability'])
    valid_driver['ridge_score'] = model.predict(valid_matrix)
    test_driver['ridge_score'] = model.predict(test_matrix)
    perform = train_data[['item_id','deal_probability']].merge(valid_driver, on='item_id')
    print('rmse:', np.sqrt(mean_squared_error(perform['deal_probability'], perform['ridge_score'])))
    del perform
    return valid_driver, test_driver

valid_1, test_1 = execute(1)
valid_2, test_2 = execute(2)
valid_3, test_3 = execute(3)
valid_4, test_4 = execute(4)
valid_5, test_5 = execute(5)
valid_scores = valid_1.append(valid_2).append(valid_3).append(valid_4).append(valid_5)
test_scores = test_1.append(test_2).append(test_3).append(test_4).append(test_5)
test_scores = test_scores.groupby('item_id')['ridge_score'].mean().reset_index()

full_scores = valid_scores.append(test_scores)
print('scores:', valid_scores.shape, test_scores.shape, full_scores.shape)
full_scores = full_scores.rename(columns={'ridge_score':'ridge_score_2'})
perform = train_data[['item_id','deal_probability']].merge(valid_scores, on='item_id')
np.sqrt(mean_squared_error(perform['deal_probability'], perform['ridge_score']))
full_scores.to_csv('../data/data/features/ridge/ridge_2.csv', index=False)