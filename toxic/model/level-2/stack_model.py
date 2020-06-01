import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import lightgbm as lgb
from functools import reduce

train_stack = pd.read_csv('../data/train_stack_scores.csv')
train_labels = pd.read_csv('../data/download/train.csv').drop('comment_text', axis=1)
train_labels = train_stack[['id']].merge(train_labels, on='id').reset_index(drop=True)
test_stack = pd.read_csv('../data/test_stack_scores.csv')
print('data:', train_labels.shape, train_stack.shape, test_stack.shape)
train_ids = train_stack[['id']].copy()
test_ids = test_stack[['id']].copy()

params = {}
params['max_depth'] = 4
params['metric'] = 'auc'
params['n_estimators'] = 200
params['num_leaves'] = 12
params['boosting_type'] = 'gbdt'
params['learning_rate'] = 0.075
params['bagging_fraction'] = 0.8
params['bagging_freq'] = 5
params['reg_lambda'] = 0.2   

folds = KFold(random_state=2017, n_splits=10)

valid_scores = []
valid_labels = []
test_scores = []

for train_idx, valid_idx in folds.split(train_ids):
    train_idx = train_ids.iloc[train_idx]['id']
    valid_idx = train_ids.iloc[valid_idx]['id']
    X_train = train_stack[train_stack['id'].isin(train_idx.values)].drop('id',axis=1)
    X_valid = train_stack[train_stack['id'].isin(valid_idx.values)].drop('id',axis=1)
    X_score = test_stack.drop('id',axis=1).copy()
    y_train = train_labels[train_labels['id'].isin(train_idx.values)].drop('id',axis=1)
    y_valid = train_labels[train_labels['id'].isin(valid_idx.values)].drop('id',axis=1)
    valid_score = np.zeros_like(y_valid)
    valid_score = pd.DataFrame(valid_score)
    test_score = np.zeros([X_score.shape[0],6])
    test_score = pd.DataFrame(test_score)
    valid_label = np.zeros_like(y_valid)
    valid_label = pd.DataFrame(valid_label)
    valid_score.columns = train_labels.columns[1:]
    valid_label.columns = train_labels.columns[1:]
    test_score.columns = train_labels.columns[1:]
    for label in train_labels.columns[1:]:
        _X_train = X_train[['rnn_' + label,'nbsvm_' + label,'logreg_' + label,'ftrl_' + label]]
        _X_valid = X_valid[['rnn_' + label,'nbsvm_' + label,'logreg_' + label,'ftrl_' + label]]
        _X_score = X_score[['rnn_' + label,'nbsvm_' + label,'logreg_' + label,'ftrl_' + label]]
        _y_train = y_train[label]
        _y_valid = y_valid[label]
        model = lgb.LGBMClassifier(**params)
        model.fit(_X_train, _y_train.values)
        valid_score[label] = model.predict_proba(_X_valid)[:,1]
        valid_score[label] = model.predict_proba(_X_valid)[:,1]
        test_score[label] = model.predict_proba(_X_score)[:,1]
        valid_label[label] = y_valid[label].values
    valid_label['id'] = valid_idx
    valid_score['id'] = valid_idx
    test_score = test_ids.join(test_score)
    valid_scores.append(valid_score)
    test_scores.append(test_score)
    valid_labels.append(valid_label)
    
valid_scores = reduce(lambda x,y : x.append(y), valid_scores)
test_scores = reduce(lambda x,y : x.append(y), test_scores)
valid_labels = reduce(lambda x,y : x.append(y), valid_labels)
overall = 0.
for label in train_labels.columns[1:]:
    fpr,tpr,threshold = roc_curve(valid_labels[label], valid_scores[label])
    overall += auc(fpr,tpr)
    print(label,':',auc(fpr,tpr))
print('overall:', overall/6)

test_scores = test_scores.groupby('id').mean().reset_index()
test_scores.to_csv('../data/test_lgb_stack.csv', index=False)
test_scores.shape

valid_scores = valid_scores[test_scores.columns]
valid_scores.to_csv('../data/train_lgb_stack.csv', index=False)
valid_scores.shape

stack = pd.read_csv('../data/train_lgb_stack.csv')
blend = pd.read_csv('../data/train_stack_scores.csv')
blend = blend[['id'] + [x for x in blend.columns if 'rnn' in x]]
blend.columns = stack.columns
score = stack.append(blend).groupby('id').mean().reset_index()
labels = valid_labels.copy()
labels = labels[test_scores.columns]

overall = 0.
for label in train_labels.columns[1:]:
    fpr,tpr,threshold = roc_curve(labels[label], score[label])
    overall += auc(fpr,tpr)
    print(label,':',auc(fpr,tpr))
print('overall:', overall/6)

score.to_csv('../data/train_lgb_rnn.csv', index=False)

stack = pd.read_csv('../data/test_lgb_stack.csv')
blend = pd.read_csv('../data/test_stack_scores.csv')
blend = blend[['id'] + [x for x in blend.columns if 'rnn' in x]]
blend.columns = stack.columns
score = stack.append(blend).groupby('id').mean().reset_index()
score.to_csv('../data/test_lgb_rnn.csv', index=False)

blend_1 = pd.read_csv('../data/other/stack/masj_stacking.csv')
blend_2 = pd.read_csv('../data/test_lgb_rnn.csv')
blend = blend_1.append(blend_2).groupby('id').mean().reset_index()
blend.to_csv('../data/test_lgb_rnn_masj.csv', index=False)

foreign = pd.read_csv('../data/download/test.csv')
function = lambda x : min([c not in x.lower() for c in "abcdefghijklmnopqrst"]) and len(x) > 0
foreign['mark'] = foreign.comment_text.map(funciton)
foreign = foreign[['id','mark']]

score = pd.read_csv('../data/test_lgb_rnn_masj.csv')
score = foreign.merge(score, on='id')
score.loc[score['mark'] == True,'toxic'] = 0.
score.loc[score['mark'] == True,'severe_toxic'] = 0.
score.loc[score['mark'] == True,'obscene'] = 0.
score.loc[score['mark'] == True,'threat'] = 0.
score.loc[score['mark'] == True,'insult'] = 0.
score.loc[score['mark'] == True,'identity_hate'] = 0.
score = score.drop('mark',axis=1)
score.to_csv('../data/test_lgb_rnn_masj_pp.csv', index=False)

score = pd.read_csv('../data/test_lgb_rnn.csv')
score = foreign.merge(score, on='id')
score.loc[score['mark'] == True,'toxic'] = 0.
score.loc[score['mark'] == True,'severe_toxic'] = 0.
score.loc[score['mark'] == True,'obscene'] = 0.
score.loc[score['mark'] == True,'threat'] = 0.
score.loc[score['mark'] == True,'insult'] = 0.
score.loc[score['mark'] == True,'identity_hate'] = 0.
score = score.drop('mark',axis=1)
score.to_csv('../data/test_lgb_rnn_pp.csv', index=False)