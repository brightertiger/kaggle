import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

model_1 = pd.read_csv('../data/score/lightgbm_1.csv').rename(columns={'score':'score_1'})
model_2 = pd.read_csv('../data/score/lightgbm_2.csv').rename(columns={'score':'score_2'})
model_3 = pd.read_csv('../data/score/lightgbm_3.csv').rename(columns={'score':'score_3'})
model_4 = pd.read_csv('../data/score/lightgbm_4.csv').rename(columns={'score':'score_4'})
model_5 = pd.read_csv('../data/score/lightgbm_5.csv').rename(columns={'score':'score_5'})
model_6 = pd.read_csv('../data/score/lightgbm_6.csv').rename(columns={'score':'score_6'})

data = model_1.merge(model_2, on=['click_id','is_attributed'])
data = data.merge(model_3, on=['click_id','is_attributed'])
data = data.merge(model_4, on=['click_id','is_attributed'])
data = data.merge(model_5, on=['click_id','is_attributed'])
data = data.merge(model_6, on=['click_id','is_attributed'])

data[['score_1','score_2','score_3','score_4','score_5','score_6']].corr()

model = LogisticRegression(tol=1e-5, C=0.5)
model.fit(data[['score_1','score_2','score_3','score_4','score_5','score_6']],data['is_attributed'])

data['score'] = 2.*data['score_1'] + 0.5*data['score_2'] + 3.*data['score_3'] + 1.*data['score_4']
data['score'] +=  3.*data['score_5'] + 1.5*data['score_6']
fpr,tpr,threshold = roc_curve(data['is_attributed'], data['score'])
auc(fpr,tpr)

model_1 = pd.read_csv('../data/full/lightgbm_1.csv').rename(columns={'is_attributed':'score_1'})
model_2 = pd.read_csv('../data/full/lightgbm_2.csv').rename(columns={'is_attributed':'score_2'})
model_3 = pd.read_csv('../data/full/lightgbm_3.csv').rename(columns={'is_attributed':'score_3'})
model_4 = pd.read_csv('../data/full/lightgbm_4.csv').rename(columns={'is_attributed':'score_4'})
model_5 = pd.read_csv('../data/full/lightgbm_5.csv').rename(columns={'is_attributed':'score_5'})
model_6 = pd.read_csv('../data/full/lightgbm_6.csv').rename(columns={'is_attributed':'score_6'})

data = model_1.merge(model_2, on=['click_id'])
data = data.merge(model_3, on=['click_id'])
data = data.merge(model_4, on=['click_id'])
data = data.merge(model_5, on=['click_id'])
data = data.merge(model_6, on=['click_id'])

data[['score_1','score_2','score_3','score_4','score_5','score_6']].corr()

data['is_attributed'] = 2.*data['score_1'] + 0.5*data['score_2'] + 3.*data['score_3'] + 1.*data['score_4']
data['is_attributed'] +=  3.*data['score_5'] + 1.5*data['score_6']
data['is_attributed'] = data['is_attributed'] / data['is_attributed'].max()
data[['click_id','is_attributed']].to_csv('../data/full/blend.csv', index=False)

model_1 = pd.read_csv('../data/submit/lightgbm_1.csv').rename(columns={'is_attributed':'score_1'})
model_2 = pd.read_csv('../data/submit/lightgbm_2.csv').rename(columns={'is_attributed':'score_2'})
model_3 = pd.read_csv('../data/submit/lightgbm_3.csv').rename(columns={'is_attributed':'score_3'})
model_4 = pd.read_csv('../data/submit/lightgbm_4.csv').rename(columns={'is_attributed':'score_4'})
model_5 = pd.read_csv('../data/submit/lightgbm_5.csv').rename(columns={'is_attributed':'score_5'})
model_6 = pd.read_csv('../data/submit/lightgbm_6.csv').rename(columns={'is_attributed':'score_6'})

data = model_1.merge(model_2, on=['click_id'])
data = data.merge(model_3, on=['click_id'])
data = data.merge(model_4, on=['click_id'])
data = data.merge(model_5, on=['click_id'])
data = data.merge(model_6, on=['click_id'])

data[['score_1','score_2','score_3','score_4','score_5','score_6']].corr()

data['is_attributed'] = 2.*data['score_1'] + 0.5*data['score_2'] + 3.*data['score_3'] + 1.*data['score_4']
data['is_attributed'] +=  3.*data['score_5'] + 1.5*data['score_6']
data['is_attributed'] = data['is_attributed'] / data['is_attributed'].max()
data[['click_id','is_attributed']].to_csv('../data/submit/blend.csv', index=False)

file_1 = pd.read_csv('../data/submit/blend.csv').sort_values(by='click_id')
file_2 = pd.read_csv('../data/full/blend.csv').sort_values(by='click_id')
file_1['is_attributed'] = 0.20* file_1['is_attributed']
file_2['is_attributed'] = 0.80* file_2['is_attributed']
file = file_1.append(file_2)
file = file.groupby('click_id')['is_attributed'].sum().reset_index()
file['is_attributed'] = file['is_attributed'] / file['is_attributed'].max()
file[['click_id','is_attributed']].to_csv('../data/blend.csv', index=False)