import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, log_loss
pd.options.mode.chained_assignment = None

model = xgb.Booster({'nthread':31})
model.load_model('../data/model/xgb_binary.model')

dependent = pd.read_csv('../data/model/dependent/dependent_n.csv')
independent = pd.read_csv('../data/model/independent/independent_n.csv')
print(dependent.shape, independent.shape)
data = dependent.merge(independent, on=['user_id','product_id','eval_set'], how='inner')
del dependent, independent
data = data[data['eval_set'] != 'train']
matrix = xgb.DMatrix(data = data.iloc[:,4:], label = data.iloc[:,3])

data = data[['user_id','product_id','reordered','eval_set']]
predict = model.predict(matrix)
data['score'] = predict

y_true = data[data['eval_set'] == 'valid']['reordered']
y_pred = data[data['eval_set'] == 'valid']['score']
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
print(2*auc(fpr,tpr) - 1, log_loss(y_true, y_pred))
data.to_csv('../data/model/score/score_n.csv', index=False)