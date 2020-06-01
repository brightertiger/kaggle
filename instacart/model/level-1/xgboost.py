import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, auc

prod_vecs = pd.read_csv('../data/gensim/prodvecs.csv')
user_vecs = pd.read_csv('../data/gensim/uservecs.csv')
train_data = pd.read_csv('../data/model/dependent/dependent_n_2.csv')
valid_data = pd.read_csv('../data/model/dependent/dependent_n_1.csv')

train_data = train_data.merge(prod_vecs, on='product_id', how='inner')
train_data = train_data.merge(user_vecs, on='user_id', how='inner')
valid_data = valid_data.merge(prod_vecs, on='product_id', how='inner')
valid_data = valid_data.merge(user_vecs, on='user_id', how='inner')

train_matrix = xgb.DMatrix(label=train_data['reordered'], data=train_data.iloc[:,4:])
valid_matrix = xgb.DMatrix(label=valid_data['reordered'], data=valid_data.iloc[:,4:])

del train_data
del valid_data

params = {}
params['booster'] = 'gbtree'
params['nthread'] = 6
params['eta'] = 0.1
params['max_depth'] = 12
params['subsample'] = 0.75
params['colsample_bytree'] = 1.0
params['colsample_bylevel'] = 0.9
params['objective'] = 'binary:logistic'
params['base_score'] = 0.10
params['eval_metric'] = 'auc'
params['seed'] = 108

model_params = {}
model_params['params'] = params
model_params['num_boost_round'] = 400
model_params['dtrain'] = train_matrix
model_params['evals'] = [(train_matrix, 'train'), (valid_matrix, 'valid')]
model_params['verbose_eval'] = 100

model = xgb.train(**model_params)
model.save_model('../data/gensim/xgb.model')

del train_matrix
del valid_matrix

model = xgb.Booster({'nthread':4})
model.load_model('../data/gensim/xgb.model')

score_data = pd.read_csv('../data/model/dependent/dependent_n_1.csv')
score_data = score_data.merge(prod_vecs, on='product_id', how='inner')
score_data = score_data.merge(user_vecs, on='user_id', how='inner')
score_matrix = xgb.DMatrix(label=score_data['reordered'], data=score_data.iloc[:,4:])
predict = model.predict(score_matrix)
score_data['xgb_w2v_score'] = predict
del score_matrix
fpr, tpr, thresholds = roc_curve(score_data['reordered'],score_data['xgb_w2v_score'])
print(round(100*(2*auc(fpr,tpr) - 1),2))
score_data = score_data[['user_id','product_id','xgb_w2v_score']]
score_data = score_data.to_csv('../data/gensim/xgb_w2_score_n_1.csv', index=False)
del score_data

score_data = pd.read_csv('../data/model/dependent/dependent_n.csv')
score_data = score_data.merge(prod_vecs, on='product_id', how='inner')
score_data = score_data.merge(user_vecs, on='user_id', how='inner')
score_matrix = xgb.DMatrix(label=score_data['reordered'], data=score_data.iloc[:,4:])
predict = model.predict(score_matrix)
score_data['xgb_w2v_score'] = predict
del score_matrix
check = score_data[score_data['eval_set'] != 'test']
fpr, tpr, thresholds = roc_curve(check['reordered'],check['xgb_w2v_score'])
print(round(100*(2*auc(fpr,tpr) - 1),2))
score_data = score_data[['user_id','product_id','xgb_w2v_score']]
score_data = score_data.to_csv('../data/gensim/xgb_w2_score_n.csv', index=False)
del score_data