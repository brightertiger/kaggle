import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

text_feats = pd.read_csv('../../data/spooky-author/data/train_text_feats.csv')
nb_score = pd.read_csv('../../data/spooky-author/data/train_nb_score.csv').drop(['id', 'author'], axis=1)
nb_feats = pd.read_csv('../../data/spooky-author/data/train_nb_feats.csv')
nn_score = pd.read_csv('../../data/spooky-author/data/train_nn_score.csv')
nn_score['nn_prob'] = np.max(np.hstack([nn_score[['k1']], nn_score[['k2']], nn_score[['k3']]]), axis=1)
nn_score = nn_score[['id','keras','nn_prob']]
lstm_score = pd.read_csv('../../data/spooky-author/data/train_lstm_score.csv')
lstm_score['lstm_prob'] = np.max(np.hstack([lstm_score[['l1']], lstm_score[['l2']], lstm_score[['l3']]]), axis=1)
lstm_score = lstm_score[['id','lstm','lstm_prob']]
train_data = text_feats.join(nb_feats)
train_data = train_data.join(nb_score)
train_data = train_data.merge(nn_score, on='id')
train_data = train_data.merge(lstm_score, on='id')
train_data['agree'] = 1.* np.equal(train_data['keras'], train_data['lstm'])
dependent = pd.read_csv('../../data/spooky-author/download/train.csv', usecols=['id','author'])
mapper = {'EAP':0, 'HPL':1, 'MWS':2}
dependent['author'] = dependent['author'].map(lambda x : mapper[x])
train_data = dependent.merge(train_data, on='id').drop('id', axis=1)
print('data shapes:', train_data.shape)
train_matrix = xgb.DMatrix(data = train_data.iloc[:,1:], label = train_data['author'])

booster = {}
booster['booster'] = 'gbtree'
booster['nthread'] = 7
booster['max_depth'] = 4
booster['min_child_weight'] = 1
booster['subsample'] = 0.75
booster['colsample_bytree'] = 1.0
booster['colsample_bylevel'] = 0.9
booster['lambda'] = 2.0
booster['alpha'] = 1.0
booster['objective'] = 'multi:softprob'
booster['eval_metric'] = ['mlogloss']
booster['num_class'] = 3
booster['seed'] = 2017

params = {}
params['params'] = booster
params['dtrain'] = train_matrix
params['num_boost_round'] = 2000
params['folds'] =  KFold(n_splits=5, random_state=2017, shuffle=True).split(train_data)
params['early_stopping_rounds'] = 50
params['verbose_eval'] = 100
params['show_stdv'] = False
params['callbacks'] = [xgb.callback.reset_learning_rate([0.05] * 2000)]

model = xgb.cv(**params)

params = {}
params['params'] = booster
params['dtrain'] = train_matrix
params['num_boost_round'] = 500
params['verbose_eval'] = 200
params['callbacks'] = [xgb.callback.reset_learning_rate([0.05] * 500)]
model = xgb.train(**params)

model.save_model('../../data/spooky-author/data/xgb_model')

sorted(model.get_fscore().items(), key=lambda x : x[1], reverse=True)[:20]
sorted(model.get_fscore().items(), key=lambda x : x[1], reverse=False)[:10]