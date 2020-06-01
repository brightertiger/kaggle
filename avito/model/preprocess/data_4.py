import pandas as pd
import numpy as np
import string
import re
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from stop_words import get_stop_words
stop_words = get_stop_words('russian')

idx = ['item_id','user_id']
encode = ['region','city','param_1','param_2','param_3','user_type']
encode += ['parent_category_name','category_name', 'image_top_1']
numeric = ['price','item_seq_number']
text = ['title', 'description']
features = ['count_1', 'count_3', 'count_4','count_5']
features += ['relative_1','relative_2','relative_5','relative_6']
features += ['renewal_1', 'avg_days_up_user']
columns = numeric + encode  + text

def merge(path):
    global train_data, test_data
    path = '../data/data/features/' + path
    feature = pd.read_csv(path)
    train_data = train_data.merge(feature, on='item_id')
    test_data = test_data.merge(feature, on='item_id')
    return None

def label_encode(variable):
    global train_data, test_data, full_data
    encode = LabelEncoder()
    encode.fit(full_data[variable].astype(str).fillna('none'))
    train_data[variable] = encode.transform(train_data[variable].astype(str).fillna('none'))
    test_data[variable] = encode.transform(test_data[variable].astype(str).fillna('none'))
    return None 

def clean_text(text):
    text = text.lower()
    text = " ".join(map(str.strip, re.split('(\d+)',text)))
    regex = re.compile(u'[^[:alpha:]]')
    text = regex.sub(" ", text)
    text = re.sub('[' + string.punctuation + ']', ' ', text)
    text = " ".join(text.split())
    return text

def long_tail(train_text, test_text):
    train_text = list(train_text.fillna('nan').values)
    test_text = list(test_text.fillna('nan').values)
    vocab = train_text + test_text
    vocab = ' '.join(vocab)
    vocab = vocab.split()
    vocab = Counter(vocab)
    global stop_words
    vocab = [word for word, count in vocab.items() if count >= 5 and word not in stop_words]
    print('vocab size:', len(vocab))
    return set(vocab)

def remove(text, vocab):
    text = text.split()
    text = map(lambda x : x if x in vocab else 'none', text)
    text = ' '.join(text)
    return text

train_data = pd.read_csv('../data/download/train.csv', usecols = columns + idx +['deal_probability'])
test_data = pd.read_csv('../data/download/test.csv', usecols = columns + idx)
full_data = train_data.append(test_data)

train_data['text'] = train_data['description'].fillna('none')  
train_data['text'] = train_data['text'] + ' ' + train_data['title'].fillna('none')
train_data['text'] = train_data['text'] + ' ' + train_data['param_3'].fillna('none')
train_data['text'] = train_data['text'] + ' ' + train_data['param_2'].fillna('none')
train_data['text'] = train_data['text'] + ' ' + train_data['param_1'].fillna('none')

test_data['text'] = test_data['description'].fillna('none')  
test_data['text'] = test_data['text'] + ' ' + test_data['title'].fillna('none')
test_data['text'] = test_data['text'] + ' ' + test_data['param_3'].fillna('none')
test_data['text'] = test_data['text'] + ' ' + test_data['param_2'].fillna('none')
test_data['text'] = test_data['text'] + ' ' + test_data['param_1'].fillna('none')

train_data = train_data.drop(['title','description'], axis=1)
test_data = test_data.drop(['title','description'], axis=1)

label_encode('region')
label_encode('city')
label_encode('param_1')
label_encode('param_2')
label_encode('param_3')
label_encode('user_type')
label_encode('parent_category_name')
label_encode('category_name')
label_encode('image_top_1')

merge('count/count_1.csv')
merge('count/count_3.csv')
merge('count/count_4.csv')
merge('count/count_5.csv')
merge('time/renew_1.csv')
merge('relative/relative_1.csv')
merge('relative/relative_2.csv')
merge('relative/relative_5.csv')
merge('relative/relative_6.csv')

user = pd.read_csv('../data/data/features/user_features.csv')
train_data = train_data.merge(user, on='user_id', how='left').drop('user_id', axis=1)
test_data = test_data.merge(user, on='user_id', how='left').drop('user_id', axis=1)

for feat in list(user.columns)[1:]:
    train_data[feat] = train_data[feat].fillna(-1)
    test_data[feat] = test_data[feat].fillna(-1)

idx.remove('user_id')
columns.remove('title')
columns.remove('description')

train_data = train_data[idx + ['deal_probability'] + features + columns + ['text']]
test_data = test_data[idx + features + columns + ['text']]

def summary(data, feature):
    summary = []
    summary += [data[feature].min()]
    summary += [data[feature].mean()]
    summary += [data[feature].max()]
    return summary

for feature in features + numeric:
    train_data[feature] = train_data[feature].fillna(0)
    test_data[feature] = test_data[feature].fillna(0)
    if feature in numeric:
        train_data[feature] = np.log(1 + train_data[feature])
        test_data[feature] = np.log(1 + test_data[feature])
    clip = train_data[feature].quantile(0.99)
    train_data[feature] = train_data[feature].clip_upper(clip)
    test_data[feature] = test_data[feature].clip_upper(clip)
    mean, std = train_data[feature].mean(), train_data[feature].std()
    train_data[feature] = (train_data[feature] - mean) / std
    test_data[feature] = (test_data[feature] - mean) / std
    print('feature:', feature)
    print(summary(train_data, feature))
    print(summary(test_data, feature))

for feature in encode:
    maximum = max(train_data[feature].max(), test_data[feature].max())
    print('dimension:', feature, maximum + 1)


vocab = long_tail(train_data['text'], test_data['text'])

train_data['text'] = train_data['text'].map(lambda x : remove(x, vocab))
test_data['text'] = test_data['text'].map(lambda x : remove(x, vocab))

train_data.to_csv('../data/data/neural/train_data.csv', index=False)
test_data.to_csv('../data/data/neural/test_data.csv', index=False)

for i in range(5):
    i += 1
    train_idx = pd.read_csv('../data/data/files/train_{}.csv'.format(i))
    valid_idx = pd.read_csv('../data/data/files/valid_{}.csv'.format(i))
    train_sub = train_idx.merge(train_data, on='item_id')
    valid_sub = valid_idx.merge(train_data, on='item_id')
    print('dataset:', train_sub.shape, valid_sub.shape)
    train_sub.to_csv('../data/data/neural/dataset/train_{}.csv'.format(i), index=False)
    valid_sub.to_csv('../data/data/neural/dataset/valid_{}.csv'.format(i), index=False)