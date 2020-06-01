import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from stop_words import get_stop_words
from multiprocessing import Pool
stop_words = get_stop_words('russian')

def clean_text(text):
    text = text.lower()
    text = " ".join(map(str.strip, re.split('(\d+)',text)))
    regex = re.compile(u'[^[:alpha:]]')
    text = regex.sub(" ", text)
    text = re.sub('[' + string.punctuation + ']', ' ', text)
    text = " ".join(text.split())
    return text

def long_tail(text):
    vocab = ' '.join(vocab)
    vocab = vocab.split()
    vocab = Counter(vocab)
    global stop_words
    vocab = [word for word, count in vocab.items() if count >= 5 and word not in stop_words]
    print('vocab size:', len(vocab))
    return set(vocab)

def remove(text, vocab):
    text = text.split()
    text = [x for x in text if x in vocab]
    text = ' '.join(text)
    return text

columns = ['title','description']

data_1 = pd.read_csv('../data/download/train.csv', usecols = columns)
data_2 = pd.read_csv('../data/download/train_active.csv', usecols = columns)
data_3 = pd.read_csv('../data/download/test.csv', usecols = columns)
data_4 = pd.read_csv('../data/download/test_active.csv', usecols = columns)
data = data_1.append(data_2).append(data_3).append(data_4)
data = data.reset_index(drop=True)
del data_1, data_2, data_3, data_4

data['text'] = data['description'].fillna('none') + ' ' + data['title'].fillna('none')
data = data.drop(['description','title'], axis=1)
data['text'] = data['text'].map(lambda x : clean_text(x))
data[['text']].to_csv('../data/data/neural/embed/text.csv', sep='\t', index=False, header=False)