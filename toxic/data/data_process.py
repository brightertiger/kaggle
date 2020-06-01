import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import KFold

def clean_text(text):
    text = text.lower()
    ipaddress = re.findall( r'[0-9]+(?:\.[0-9]+){3}', text)
    for ip in ipaddress:
        text = text.replace(ip,' ')
    text = text.replace('\n', ' ')
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', text)
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip().lower()

labels = ['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']
text = ['id','comment_text']

train_data = pd.read_csv('../data/download/train.csv').fillna('nan')
train_data['comment_text'] = train_data['comment_text'].map(lambda x : clean_text(x))

test_data = pd.read_csv('../data/download/test.csv').fillna('nan')
test_data['comment_text'] = test_data['comment_text'].map(lambda x : clean_text(x))

comments = train_data['id'].unique()
folds = KFold(n_splits=10, shuffle=True, random_state=2017)
fold = 1
source = 'source_1'

for train_idx, test_idx in folds.split(comments):
    train_idx = comments[train_idx]
    test_idx = comments[test_idx]
    X_train = train_data[train_data['id'].isin(train_idx)][text]
    y_train = train_data[train_data['id'].isin(train_idx)][labels]
    X_test = train_data[train_data['id'].isin(test_idx)][text]
    y_test = train_data[train_data['id'].isin(test_idx)][labels]
    print('train data:', X_train.shape, y_train.shape)
    print('test data:', X_test.shape, y_test.shape)
    path = '../data/data/{}/train/'.format(source)
    X_train.to_csv(path + 'train_data_{}.csv'.format(fold), index=False)
    y_train.to_csv(path + 'train_labels_{}.csv'.format(fold), index=False)
    X_test.to_csv(path + 'test_data_{}.csv'.format(fold), index=False)
    y_test.to_csv(path + 'test_labels_{}.csv'.format(fold), index=False)
    fold += 1

source = 'source_1'
path = '../data/data/{}/score/'.format(source)
test_data.to_csv(path + 'score_data.csv', index=False)
print('test data:', test_data.shape)

def clean_text(text):
    ipaddress = re.findall( r'[0-9]+(?:\.[0-9]+){3}', text)
    for ip in ipaddress:
        text = text.replace(ip,' ')
    text = text.replace('\n', ' ')
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', text)
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

labels = ['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']
text = ['id','comment_text']

train_data = pd.read_csv('../data/download/train.csv').fillna('nan')
train_data['comment_text'] = train_data['comment_text'].map(lambda x : clean_text(x))

test_data = pd.read_csv('../data/download/test.csv').fillna('nan')
test_data['comment_text'] = test_data['comment_text'].map(lambda x : clean_text(x))

comments = train_data['id'].unique()
folds = KFold(n_splits=10, shuffle=True, random_state=2017)
fold = 1
source = 'source_2'

for train_idx, test_idx in folds.split(comments):
    train_idx = comments[train_idx]
    test_idx = comments[test_idx]
    X_train = train_data[train_data['id'].isin(train_idx)][text]
    y_train = train_data[train_data['id'].isin(train_idx)][labels]
    X_test = train_data[train_data['id'].isin(test_idx)][text]
    y_test = train_data[train_data['id'].isin(test_idx)][labels]
    print('train data:', X_train.shape, y_train.shape)
    print('test data:', X_test.shape, y_test.shape)
    path = '../data/data/{}/train/'.format(source)
    X_train.to_csv(path + 'train_data_{}.csv'.format(fold), index=False)
    y_train.to_csv(path + 'train_labels_{}.csv'.format(fold), index=False)
    X_test.to_csv(path + 'test_data_{}.csv'.format(fold), index=False)
    y_test.to_csv(path + 'test_labels_{}.csv'.format(fold), index=False)
    fold += 1

source = 'source_2'
path = '../data/data/{}/score/'.format(source)
test_data.to_csv(path + 'score_data.csv', index=False)
print('test data:', test_data.shape)

import regex as re
FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps> "

def tokenize(text):
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    text = re_sub(r"([A-Z]){2,}", allcaps)
    punct = re.compile('[%s]' % re.escape(string.punctuation.replace('<','').replace('>','')))
    text = punct.sub(' ', text)
    text = re.sub('\s+', ' ', text)
    return text.lower()

labels = ['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']
text = ['id','comment_text']

train_data = pd.read_csv('../data/download/train.csv').fillna('nan')
train_data['comment_text'] = train_data['comment_text'].map(lambda x : tokenize(x))

test_data = pd.read_csv('../data/download/test.csv').fillna('nan')
test_data['comment_text'] = test_data['comment_text'].map(lambda x : tokenize(x))

comments = train_data['id'].unique()
folds = KFold(n_splits=10, shuffle=True, random_state=2017)
fold = 1
source = 'source_3'

for train_idx, test_idx in folds.split(comments):
    train_idx = comments[train_idx]
    test_idx = comments[test_idx]
    X_train = train_data[train_data['id'].isin(train_idx)][text]
    y_train = train_data[train_data['id'].isin(train_idx)][labels]
    X_test = train_data[train_data['id'].isin(test_idx)][text]
    y_test = train_data[train_data['id'].isin(test_idx)][labels]
    print('train data:', X_train.shape, y_train.shape)
    print('test data:', X_test.shape, y_test.shape)
    path = '../data/data/{}/train/'.format(source)
    X_train.to_csv(path + 'train_data_{}.csv'.format(fold), index=False)
    y_train.to_csv(path + 'train_labels_{}.csv'.format(fold), index=False)
    X_test.to_csv(path + 'test_data_{}.csv'.format(fold), index=False)
    y_test.to_csv(path + 'test_labels_{}.csv'.format(fold), index=False)
    fold += 1

source = 'source_3'
path = '../data/data/{}/score/'.format(source)
test_data.to_csv(path + 'score_data.csv', index=False)
print('test data:', test_data.shape)

import nltk

def clean_text(text):
    ipaddress = re.findall( r'[0-9]+(?:\.[0-9]+){3}', text)
    for ip in ipaddress:
        text = text.replace(ip,' ')
    text = text.replace('\n', ' ')
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = nltk.word_tokenize(text)
    return ' '.join([x.strip() for x in text])

labels = ['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']
text = ['id','comment_text']

train_data = pd.read_csv('../data/download/train.csv').fillna('nan')
train_data['comment_text'] = train_data['comment_text'].map(lambda x : clean_text(x))

test_data = pd.read_csv('../data/download/test.csv').fillna('nan')
test_data['comment_text'] = test_data['comment_text'].map(lambda x : clean_text(x))

comments = train_data['id'].unique()
folds = KFold(n_splits=10, shuffle=True, random_state=2017)
fold = 1
source = 'source_4'

for train_idx, test_idx in folds.split(comments):
    train_idx = comments[train_idx]
    test_idx = comments[test_idx]
    X_train = train_data[train_data['id'].isin(train_idx)][text]
    y_train = train_data[train_data['id'].isin(train_idx)][labels]
    X_test = train_data[train_data['id'].isin(test_idx)][text]
    y_test = train_data[train_data['id'].isin(test_idx)][labels]
    print('train data:', X_train.shape, y_train.shape)
    print('test data:', X_test.shape, y_test.shape)
    path = '../data/data/{}/train/'.format(source)
    X_train.to_csv(path + 'train_data_{}.csv'.format(fold), index=False)
    y_train.to_csv(path + 'train_labels_{}.csv'.format(fold), index=False)
    X_test.to_csv(path + 'test_data_{}.csv'.format(fold), index=False)
    y_test.to_csv(path + 'test_labels_{}.csv'.format(fold), index=False)
    fold += 1

source = 'source_4'
path = '../data/data/{}/score/'.format(source)
test_data.to_csv(path + 'score_data.csv', index=False)
print('test data:', test_data.shape)

labels = ['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']
text = ['id','comment_text']

train_data = pd.read_csv('../data/download/train_preprocessed.csv').fillna('nan')
test_data = pd.read_csv('../data/download/test_preprocessed.csv').fillna('nan')

comments = train_data['id'].unique()
folds = KFold(n_splits=10, shuffle=True, random_state=2017)
fold = 1
source = 'source_5'

for train_idx, test_idx in folds.split(comments):
    train_idx = comments[train_idx]
    test_idx = comments[test_idx]
    X_train = train_data[train_data['id'].isin(train_idx)][text]
    y_train = train_data[train_data['id'].isin(train_idx)][labels]
    X_test = train_data[train_data['id'].isin(test_idx)][text]
    y_test = train_data[train_data['id'].isin(test_idx)][labels]
    print('train data:', X_train.shape, y_train.shape)
    print('test data:', X_test.shape, y_test.shape)
    path = '../data/data/{}/train/'.format(source)
    X_train.to_csv(path + 'train_data_{}.csv'.format(fold), index=False)
    y_train.to_csv(path + 'train_labels_{}.csv'.format(fold), index=False)
    X_test.to_csv(path + 'test_data_{}.csv'.format(fold), index=False)
    y_test.to_csv(path + 'test_labels_{}.csv'.format(fold), index=False)
    fold += 1

source = 'source_5'
path = '../data/data/{}/score/'.format(source)
test_data.to_csv(path + 'score_data.csv', index=False)
print('test data:', test_data.shape)

labels = ['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']
text = ['id','comment_text']

train_data = pd.read_csv('../data/download/train_preprocessed.csv').fillna('nan')
test_data = pd.read_csv('../data/download/test_preprocessed.csv').fillna('nan')

comments = train_data['id'].unique()
folds = KFold(n_splits=10, shuffle=True, random_state=2017)
fold = 1
source = 'source_6'

for train_idx, test_idx in folds.split(comments):
    train_idx = comments[train_idx]
    test_idx = comments[test_idx]
    X_train = train_data[train_data['id'].isin(train_idx)][text]
    y_train = train_data[train_data['id'].isin(train_idx)][labels]
    X_test = train_data[train_data['id'].isin(test_idx)][text]
    y_test = train_data[train_data['id'].isin(test_idx)][labels]
    print('train data:', X_train.shape, y_train.shape)
    print('test data:', X_test.shape, y_test.shape)
    path = '../data/data/{}/train/'.format(source)
    X_train.to_csv(path + 'train_data_{}.csv'.format(fold), index=False)
    y_train.to_csv(path + 'train_labels_{}.csv'.format(fold), index=False)
    X_test.to_csv(path + 'test_data_{}.csv'.format(fold), index=False)
    y_test.to_csv(path + 'test_labels_{}.csv'.format(fold), index=False)
    fold += 1

source = 'source_6'
path = '../data/data/{}/score/'.format(source)
test_data.to_csv(path + 'score_data.csv', index=False)
print('test data:', test_data.shape)