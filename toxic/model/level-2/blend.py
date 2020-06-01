import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
pd.options.mode.chained_assignment = None

train_score_1 = pd.read_csv('../data/model/model_1.csv')
train_score_1.columns = [train_score_1.columns[0]] + ['m1_' + x for x in train_score_1.columns[1:]]
train_score_2 = pd.read_csv('../data/model/model_2.csv')
train_score_2.columns = [train_score_2.columns[0]] + ['m2_' + x for x in train_score_2.columns[1:]]
train_score_3 = pd.read_csv('../data/model/model_3.csv')
train_score_3.columns = [train_score_3.columns[0]] + ['m3_' + x for x in train_score_3.columns[1:]]
train_score_4 = pd.read_csv('../data/model/model_4.csv')
train_score_4.columns = [train_score_4.columns[0]] + ['m4_' + x for x in train_score_4.columns[1:]]
train_score_5 = pd.read_csv('../data/model/model_5.csv')
train_score_5.columns = [train_score_5.columns[0]] + ['m5_' + x for x in train_score_5.columns[1:]]
train_score_6 = pd.read_csv('../data/model/model_6.csv')
train_score_6.columns = [train_score_6.columns[0]] + ['m6_' + x for x in train_score_6.columns[1:]]
train_score_7 = pd.read_csv('../data/model/model_7.csv')
train_score_7.columns = [train_score_7.columns[0]] + ['m7_' + x for x in train_score_7.columns[1:]]
train_score_8 = pd.read_csv('../data/model/model_8.csv')
train_score_8.columns = [train_score_8.columns[0]] + ['m8_' + x for x in train_score_8.columns[1:]]
train_score_9 = pd.read_csv('../data/model/model_9.csv')
train_score_9.columns = [train_score_9.columns[0]] + ['m9_' + x for x in train_score_9.columns[1:]]
train_score_10 = pd.read_csv('../data/model/model_10.csv')
train_score_10.columns = [train_score_10.columns[0]] + ['m10_' + x for x in train_score_10.columns[1:]]
train_score_11 = pd.read_csv('../data/model/model_11.csv')
train_score_11.columns = [train_score_11.columns[0]] + ['m11_' + x for x in train_score_11.columns[1:]]

train_labels = pd.read_csv('../data/download/train.csv').drop('comment_text', axis=1)
train_data = train_score_1.merge(train_score_2, on='id')
train_data = train_data.merge(train_score_3, on='id')
train_data = train_data.merge(train_score_4, on='id')
train_data = train_data.merge(train_score_5, on='id')
train_data = train_data.merge(train_score_6, on='id')
train_data = train_data.merge(train_score_7, on='id')
train_data = train_data.merge(train_score_8, on='id')
train_data = train_data.merge(train_score_9, on='id')
train_data = train_data.merge(train_score_10, on='id')
train_data = train_data.merge(train_score_11, on='id')

train_labels = train_labels.sort_values(by='id').reset_index(drop=True)
train_labels = train_labels.merge(train_data[['id']], on='id')
train_data = train_data.sort_values(by='id').reset_index(drop=True)
train_ids = train_labels[['id']].copy()
train_data = train_data.drop('id', axis=1)
train_labels = train_labels.drop('id', axis=1)
print('train_data:', train_data.shape, train_labels.shape)
del train_score_1, train_score_2

test_score_1 = pd.read_csv('../data/submit/model_1.csv')
test_score_1.columns = [test_score_1.columns[0]] + ['m1_' + x for x in test_score_1.columns[1:]]
test_score_2 = pd.read_csv('../data/submit/model_2.csv')
test_score_2.columns = [test_score_2.columns[0]] + ['m2_' + x for x in test_score_2.columns[1:]]
test_score_3 = pd.read_csv('../data/submit/model_3.csv')
test_score_3.columns = [test_score_3.columns[0]] + ['m3_' + x for x in test_score_3.columns[1:]]
test_score_4 = pd.read_csv('../data/submit/model_4.csv')
test_score_4.columns = [test_score_4.columns[0]] + ['m4_' + x for x in test_score_4.columns[1:]]
test_score_5 = pd.read_csv('../data/submit/model_5.csv')
test_score_5.columns = [test_score_5.columns[0]] + ['m5_' + x for x in test_score_5.columns[1:]]
test_score_6 = pd.read_csv('../data/submit/model_6.csv')
test_score_6.columns = [test_score_6.columns[0]] + ['m6_' + x for x in test_score_6.columns[1:]]
test_score_7 = pd.read_csv('../data/submit/model_7.csv')
test_score_7.columns = [test_score_7.columns[0]] + ['m7_' + x for x in test_score_7.columns[1:]]
test_score_8 = pd.read_csv('../data/submit/model_8.csv')
test_score_8.columns = [test_score_8.columns[0]] + ['m8_' + x for x in test_score_8.columns[1:]]
test_score_9 = pd.read_csv('../data/submit/model_9.csv')
test_score_9.columns = [test_score_9.columns[0]] + ['m9_' + x for x in test_score_9.columns[1:]]
test_score_10 = pd.read_csv('../data/submit/model_10.csv')
test_score_10.columns = [test_score_10.columns[0]] + ['m10_' + x for x in test_score_10.columns[1:]]
test_score_11 = pd.read_csv('../data/submit/model_11.csv')
test_score_11.columns = [test_score_11.columns[0]] + ['m11_' + x for x in test_score_11.columns[1:]]

test_data = test_score_1.merge(test_score_2, on='id')
test_data = test_data.merge(test_score_3, on='id')
test_data = test_data.merge(test_score_4, on='id')
test_data = test_data.merge(test_score_5, on='id')
test_data = test_data.merge(test_score_6, on='id')
test_data = test_data.merge(test_score_7, on='id')
test_data = test_data.merge(test_score_8, on='id')
test_data = test_data.merge(test_score_9, on='id')
test_data = test_data.merge(test_score_10, on='id')
test_data = test_data.merge(test_score_11, on='id')

test_ids = test_data[['id']].copy()
test_data = test_data.drop('id', axis=1)
print('test_data:', test_data.shape)
del test_score_1, test_score_2

overall = 0. 

def eval_metric(labels, predict):
    fpr, tpr, threshold = roc_curve(labels, predict)
    return round(auc(fpr, tpr),4)
    
def model(label):
    feats = ['m1_' + label, 'm2_' + label,'m3_' + label, 'm4_' + label, 'm5_' + label, 'm6_' + label]
    feats += ['m7_' + label, 'm8_' + label,'m9_' + label, 'm10_' + label, 'm11_' + label]
    scores = train_data[feats]
    scores[label] = scores.apply(lambda x : np.mean(x), axis=1)
    labels = train_labels[label]
    print('auc:', eval_metric(labels,scores[label]))
    model = train_ids.copy()
    model[label] = scores[label].copy()
    global overall
    overall += eval_metric(labels,scores[label])
    scores = test_data[feats]
    scores[label] = scores.apply(lambda x : np.mean(x), axis=1)
    submit = test_ids.copy()
    submit[label] = scores[label].copy()
    return [model, submit]

toxic = model('toxic')
severe_toxic = model('severe_toxic')
obscene = model('obscene')
threat = model('threat')
insult = model('insult')
identity_hate = model('identity_hate')

model = toxic[0].copy()
model = model.merge(severe_toxic[0], on='id')
model = model.merge(obscene[0], on='id')
model = model.merge(threat[0], on='id')
model = model.merge(insult[0], on='id')
model = model.merge(identity_hate[0], on='id')

submit = toxic[1].copy()
submit = submit.merge(severe_toxic[1], on='id')
submit = submit.merge(obscene[1], on='id')
submit = submit.merge(threat[1], on='id')
submit = submit.merge(insult[1], on='id')
submit = submit.merge(identity_hate[1], on='id')

model.to_csv('../data/model/simple_stack.csv', index=False)
submit.to_csv('../data/submit/simple_stack.csv', index=False)