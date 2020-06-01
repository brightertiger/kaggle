from functools import reduce
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train = pd.read_csv('../data/download/train.csv').fillna(' ')
test = pd.read_csv('../data/download/test.csv').fillna(' ')
train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

params ={}
params['analyzer'] = 'word'
params['ngram_range'] = (1,1)
params['token_pattern'] = r'\w{1,}'
params['stop_words'] = 'english'
params['strip_accents'] = 'unicode'
params['sublinear_tf'] = 1
params['max_features'] = 10000
word_vectorizer = TfidfVectorizer(**params)
word_vectorizer.fit(all_text)

params ={}
params['analyzer'] = 'char'
params['ngram_range'] = (2,6)
params['token_pattern'] = r'\w{1,}'
params['stop_words'] = 'english'
params['strip_accents'] = 'unicode'
params['sublinear_tf'] = 1
params['max_features'] = 50000
char_vectorizer = TfidfVectorizer(**params)
char_vectorizer.fit(all_text)

def model(train_labels, train_text, valid_text, test_text):
    train_word_doc = word_vectorizer.fit_transform(train_text['comment_text'].fillna('nan'))
    valid_word_doc = word_vectorizer.transform(valid_text['comment_text'].fillna('nan'))
    test_word_doc = word_vectorizer.transform(test_text['comment_text'].fillna('nan'))
    train_char_doc = char_vectorizer.fit_transform(train_text['comment_text'].fillna('nan'))
    valid_char_doc = char_vectorizer.transform(valid_text['comment_text'].fillna('nan'))
    test_char_doc = char_vectorizer.transform(test_text['comment_text'].fillna('nan'))
    train_features = hstack([train_char_doc, train_word_doc])
    valid_features = hstack([valid_char_doc, valid_word_doc])
    test_features = hstack([test_char_doc, test_word_doc])
    valid_ids = valid_text[['id']].copy()
    test_ids = test_text[['id']].copy()
    valid_score = np.zeros([valid_text.shape[0],train_labels.shape[1]-1])
    test_score = np.zeros([test_text.shape[0],train_labels.shape[1]-1])
    for idx in range(train_labels.shape[1]-1):
        model = LogisticRegression(C=0.1, solver='sag')
        model.fit(train_features, train_labels.iloc[:,idx+1].values)
        valid_score[:,idx] = model.predict_proba(valid_features)[:,1]
        test_score[:,idx] = model.predict_proba(test_features)[:,1]
    valid_score = pd.DataFrame(valid_score)
    valid_score.columns = list(train_labels.columns)[1:]
    valid_score = valid_ids.join(valid_score)
    test_score = pd.DataFrame(test_score)
    test_score.columns = list(train_labels.columns)[1:]
    test_score = test_ids.join(test_score)
    return valid_score, test_score

def execute(mode):
    train_text = pd.read_csv('../data/data/source_6/train/train_data_{}.csv'.format(mode))
    train_label = pd.read_csv('../data/data/source_6/train/train_labels_{}.csv'.format(mode))
    valid_text = pd.read_csv('../data/data/source_6/train/test_data_{}.csv'.format(mode))
    score_text = pd.read_csv('../data/data/source_6/score/score_data.csv'.format(mode))
    valid_score, test_score = model(train_label, train_text, valid_text, score_text)
    return valid_score, test_score

valid_scores = []
test_scores = []

for idx in range(9):
    temp1, temp2 = execute(idx+1)
    valid_scores.append(temp1)
    test_scores.append(temp2)
    print('{} model executed.'.format(idx))

valid_data = reduce(lambda x,y : x.append(y), valid_scores)
score_data = reduce(lambda x,y : x.append(y), test_scores)
score_data = score_data.groupby('id').mean().reset_index()
print('Cross Validation:', valid_data.shape)
print('Scoring:', score_data.shape)

valid_data.to_csv('../data/model/ftrl.csv', index=False)
score_data.to_csv('../data/submit/ftrl.csv', index=False)

labels = pd.read_csv('../data/download/train.csv')
labels = labels.drop('comment_text', axis=1)
labels = valid_data[['id']].merge(labels, on='id')
print('data:', valid_data.shape, labels.shape)

models = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
evaluate = 0.

for subset in models:
    predict = valid_data[subset]
    actual = labels[subset]
    fpr, tpr, threshold = roc_curve(actual, predict)
    metric = round(2*auc(fpr, tpr)-1, 4)
    print('label:', subset, ':', metric)
    evaluate += metric
    
print('overall:', round(evaluate/6, 4))