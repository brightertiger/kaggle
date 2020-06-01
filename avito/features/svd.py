import pandas as pd
import numpy as np
import re
import string
import stop_words
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
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
features = ['item_id','title','description','params']
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
matrix = vectorizer.fit_transform(full_data.to_dict('records'))
driver = train_data[['item_id']].append(test_data[['item_id']])
driver = driver.reset_index(drop=True)
svd = TruncatedSVD(n_components=3, random_state=2017)
decompose = svd.fit_transform(matrix)
print('variance:', sum(svd.explained_variance_ratio_), list(svd.explained_variance_ratio_))
driver['svd_1'] = decompose[:,0]
driver['svd_2'] = decompose[:,1]
driver['svd_3'] = decompose[:,2]
driver.to_csv('../data/data/features/ridge/svd.csv', index=False)