import pandas as pd
import numpy as np
from functools import reduce
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_curve, auc
import re
import string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): 
    return re_tok.sub(r' \1 ', s).split()

def probability(document, y_i, y):
    p = document[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def compute(document, labels):
    y = labels.values
    r = np.log(probability(document,1,y) / probability(document,0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = document.multiply(r)
    return m.fit(x_nb, y), r

def model(train_labels, train_text, valid_text, test_text):
    n = train_text.shape[0]
    params ={}
    params['ngram_range'] = (1,2)
    params['tokenizer'] = tokenize
    params['min_df'] = 3
    params['max_df'] = 0.9
    params['strip_accents'] = 'unicode'
    params['use_idf'] = 1
    params['smooth_idf'] = 1
    params['sublinear_tf'] = 1
    vectorize = TfidfVectorizer(**params)
    train_doc = vectorize.fit_transform(train_text['comment_text'].fillna('nan'))
    valid_doc = vectorize.transform(valid_text['comment_text'].fillna('nan'))
    test_doc = vectorize.transform(test_text['comment_text'].fillna('nan'))
    valid_ids = valid_text[['id']].copy()
    test_ids = test_text[['id']].copy()
    valid_score = np.zeros([valid_text.shape[0],train_labels.shape[1]-1])
    test_score = np.zeros([test_text.shape[0],train_labels.shape[1]-1])
    for idx in range(train_labels.shape[1]-1):
        model,result = compute(train_doc, train_labels.iloc[:,idx+1])
        valid_score[:,idx] = model.predict_proba(valid_doc.multiply(result))[:,1]
        test_score[:,idx] = model.predict_proba(test_doc.multiply(result))[:,1]
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

valid_data.to_csv('../data/model/nbsvm.csv', index=False)
score_data.to_csv('../data/submit/nbsvm.csv', index=False)

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