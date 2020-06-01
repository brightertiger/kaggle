import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from stop_words import get_stop_words
russian = ['а', 'э', 'ы', 'у', 'о', 'я', 'е', 'ё', 'ю', 'и']
english = 'abcdefghijklmnopqrstuvwxyz' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
stop_words = get_stop_words('russian')

usecols=['item_id','title']
train_actuals = pd.read_csv('../data/download/train.csv', usecols=['deal_probability']) 
train_data = pd.read_csv('../data/download/train.csv', usecols=usecols)
test_data = pd.read_csv('../data/download/test.csv', usecols=usecols)

def vowels(text):
    for char in text:
        if char in russian:
            return 1
    return 0

train_data['txt_tl_1'] = train_data['title'].fillna(' ').map(lambda x : vowels(x))
test_data['txt_tl_1'] = test_data['title'].fillna(' ').map(lambda x : vowels(x))

def engchars(text):
    count = 0
    for char in text:
        if char in english:
            count += 1
    return count

train_data['txt_tl_2'] = train_data['title'].fillna(' ').map(lambda x : engchars(x))
test_data['txt_tl_2'] = test_data['title'].fillna(' ').map(lambda x : engchars(x))

def punctuation(text):
    count = 0
    for char in text:
        if not str.isalnum(char):
            count += 1
    return count

train_data['txt_tl_3'] = train_data['title'].fillna(' ').map(lambda x : punctuation(x))
test_data['txt_tl_3'] = test_data['title'].fillna(' ').map(lambda x : punctuation(x))

train_data['txt_tl_4'] = train_data['title'].fillna(' ').map(lambda x : len(x.split()))
test_data['txt_tl_4'] = test_data['title'].fillna(' ').map(lambda x : len(x.split()))

def wordlen(text):
    total = 0
    count = 0
    for word in text.split():
        total += len(word)
        count += 1
    return total / max(count,1)

train_data['txt_tl_5'] = train_data['title'].fillna(' ').map(lambda x : wordlen(x))
test_data['txt_tl_5'] = test_data['title'].fillna(' ').map(lambda x : wordlen(x))

def numeric(text):
    count = 0
    for char in text:
        if str.isnumeric(char):
            count += 1
    return count

train_data['txt_tl_6'] = train_data['title'].fillna(' ').map(lambda x : numeric(x))
test_data['txt_tl_6'] = test_data['title'].fillna(' ').map(lambda x : numeric(x))

train_data['txt_tl_7'] = train_data['title'].fillna(' ').map(lambda x : len(x))
test_data['txt_tl_7'] = test_data['title'].fillna(' ').map(lambda x : len(x))

def upper(text):
    count = 0
    total = len(text)
    for char in text:
        if char == char.upper() and char != ' ':
            count += 1
    return count / max(total, 1)

train_data['txt_tl_8'] = train_data['title'].fillna(' ').map(lambda x : upper(x))
test_data['txt_tl_8'] = test_data['title'].fillna(' ').map(lambda x : upper(x))

def all_caps(text):
    words = text.split()
    count = 0
    total = len(words)
    for word in words:
        if word == word.upper():
            count += 1
    return count / max(total, 1)

train_data['txt_tl_9'] = train_data['title'].fillna(' ').map(lambda x : all_caps(x))
test_data['txt_tl_9'] = test_data['title'].fillna(' ').map(lambda x : all_caps(x))

def stp_wrds(text):
    words = text.lower().split()
    count = 0
    global stop_words
    for word in words:
        if word in stop_words:
            count += 1
    return count

train_data['txt_tl_10'] = train_data['title'].fillna(' ').map(lambda x : stp_wrds(x))
test_data['txt_tl_10'] = test_data['title'].fillna(' ').map(lambda x : stp_wrds(x))

vocab = list(train_data['title'].values) + list(test_data['title'].values)
vocab = ' '.join(vocab)
vocab = vocab.lower().split()
vocab = Counter(vocab)
vocab = [word for word, count in vocab.items() if count > 1000 and word not in stop_words]

def evaluate(word):
    global train_data
    dummy = train_data['title'].map(lambda x : 1 if word in x.lower() else 0)
    positive = train_actuals[dummy == 1].mean().values[0]
    negative = train_actuals[dummy == 0].mean().values[0]
    ratio_plus =  positive / negative
    ratio_minus = negative / positive
    return ratio_plus, ratio_minus

positive = []
negative = []
for word in vocab:
    ratio_plus, ratio_minus = evaluate(word)
    if ratio_plus > 1.5:
        positive += [word]
    elif ratio_minus > 1.5:
        negative += [word]

print('positive words:', len(positive))
print('negative words:', len(negative))

def count_pos(text):
    global positive
    text = text.split(' ')
    text = [x for x in text if x.lower() in positive]
    return len(text)

def count_neg(text):
    global negative
    text = text.split(' ')
    text = [x for x in text if x.lower() in negative]
    return len(text)

train_data['txt_tl_pos'] = train_data['title'].map(lambda x : count_pos(x))
test_data['txt_tl_pos'] = test_data['title'].map(lambda x : count_pos(x))

train_data['txt_tl_neg'] = train_data['title'].map(lambda x : count_neg(x))
test_data['txt_tl_neg'] = test_data['title'].map(lambda x : count_neg(x))

features = train_data.append(test_data).drop('title', axis=1)
features.to_csv('../data/data/features/nlp/title.csv', index=False)