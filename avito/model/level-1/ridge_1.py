import pandas as pd
import numpy as np
import re
import string
import stop_words
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import mean_squared_error
stopwords = set(stop_words.get_stop_words('russian'))

def clean_text(text):
    text = text.lower()
    text = " ".join(map(str.strip, re.split('(\d+)',text)))
    regex = re.compile(u'[^[:alpha:]]')
    text = regex.sub(" ", text)
    text = re.sub('[' + string.punctuation + ']', ' ', text)
    text = " ".join(text.split())
    return text

def concatenate(row):
    return ' '.join([str(row['param_1']), str(row['param_2']), str(row['param_3'])])

columns = ['item_id', 'param_1','param_2','param_3','title','description']
train_data = pd.read_csv('../data/download/train.csv', usecols=columns + ['deal_probability'])
test_data = pd.read_csv('../data/download/test.csv', usecols=columns)

train_data['title'] = train_data['title'].fillna('none').apply(lambda x: clean_text(x))
train_data['description'] = train_data['description'].fillna('none').apply(lambda x: clean_text(x))
train_data['params'] = train_data.apply(lambda row: concatenate(row),axis=1)

test_data['title'] = test_data['title'].fillna('none').apply(lambda x: clean_text(x))
test_data['description'] = test_data['description'].fillna('none').apply(lambda x: clean_text(x))
test_data['params'] = test_data.apply(lambda row: concatenate(row),axis=1)
features = ['title','description','params']
full_data = train_data[features].append(test_data[features])

def fetch(column):
    return lambda x: x[column]

params = {}
params['stop_words'] = stopwords
params['analyzer'] = 'word'
params['token_pattern'] = r'\w{1,}'
params['sublinear_tf'] = True
params['dtype'] = np.float32
params['norm'] = 'l2'
params['smooth_idf'] = False
params['ngram_range'] = (1,2)

pipe = []
pipe += [('description', TfidfVectorizer(**params, preprocessor=fetch('description'), max_features=50000))]
pipe += [('title', TfidfVectorizer(**params, preprocessor=fetch('title')))]
pipe += [('params', CountVectorizer(ngram_range=(1, 2), preprocessor=fetch('params')))]

vectorizer = FeatureUnion(pipe)
vectorizer.fit(full_data.to_dict('records'))

params = {}
params['alpha'] = 20.
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
full_scores.to_csv('../data/ta/data/features/ridge/ridge_1.csv', index=False)
perform = train_data[['item_id','deal_probability']].merge(valid_scores, on='item_id')
np.sqrt(mean_squared_error(perform['deal_probability'], perform['ridge_score']))