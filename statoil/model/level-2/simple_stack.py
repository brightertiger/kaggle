import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

model_1 = pd.read_csv('../data/model/baseline_v1.csv').rename(columns={'score':'model_1'})
model_2 = pd.read_csv('../data/model/baseline_v2.csv').rename(columns={'score':'model_2'})
model_3 = pd.read_csv('../data/model/vgg11_v1.csv').rename(columns={'score':'model_3'})
model_4 = pd.read_csv('../data/model/vgg11_v2.csv').rename(columns={'score':'model_4'})
model_5 = pd.read_csv('../data/model/vgg16_v1.csv').rename(columns={'score':'model_5'})
model_6 = pd.read_csv('../data/model/vgg16_v2.csv').rename(columns={'score':'model_6'})

train = model_1.copy()
train = train.merge(model_2, on=['id','label'])
train = train.merge(model_3, on=['id','label'])
train = train.merge(model_4, on=['id','label'])
train = train.merge(model_5, on=['id','label'])
train = train.merge(model_6, on=['id','label'])
print('train:', train.shape)
for feat in train.columns[2:]:
    print(feat, ':', log_loss(train['label'], train[feat]))

model_1 = pd.read_csv('../data/submit/baseline_v1.csv').rename(columns={'is_iceberg':'model_1'})
model_2 = pd.read_csv('../data/submit/baseline_v2.csv').rename(columns={'is_iceberg':'model_2'})
model_3 = pd.read_csv('../data/submit/vgg11_v1.csv').rename(columns={'is_iceberg':'model_3'})
model_4 = pd.read_csv('../data/submit/vgg11_v2.csv').rename(columns={'is_iceberg':'model_4'})
model_5 = pd.read_csv('../data/submit/vgg16_v1.csv').rename(columns={'is_iceberg':'model_5'})
model_6 = pd.read_csv('../data/submit/vgg16_v2.csv').rename(columns={'is_iceberg':'model_6'})

test = model_1.copy()
test = test.merge(model_2, on=['id'])
test = test.merge(model_3, on=['id'])
test = test.merge(model_4, on=['id'])
test = test.merge(model_5, on=['id'])
test = test.merge(model_6, on=['id'])
print('test:', test.shape)

def stack_func(value, low, high):
    if np.all(value < low):
        return min(value)
    elif np.all(value > high):
        return max(value)
    else:
        return np.mean(value)

scores = train[['model_1','model_2','model_3','model_4','model_5','model_6']].copy()
low_values = list(range(5, 41, 5))
high_values = list(range(60,96,5))
benchmark = 0.17

for low_ in low_values:
    for high_ in high_values:
        predict = scores.apply(lambda x : stack_func(x, low_/100,high_/100), axis=1).clip(0.001, 0.999)
        if benchmark > log_loss(train['label'], predict):
            print('low, high:', low_, high_)
            print('loss:', log_loss(train['label'], predict))

train['stack'] = scores.apply(lambda x : stack_func(x, 15/100, 95/100), axis=1).clip(0.001, 0.999)
print('loss:', log_loss(train['label'], train['stack']))
train.to_csv('../data/train_scores.csv', index=False)

scores = test[['model_1','model_2','model_3','model_4','model_5','model_6']].copy()
test['stack'] = scores.apply(lambda x : stack_func(x, 15/100, 95/100), axis=1).clip(0.001, 0.999)
test.to_csv('../data/test_scores.csv', index=False)
submit = test[['id','stack']].copy()
submit = submit.rename(columns={'stack':'is_iceberg'})
submit.to_csv('../data/submit/stacked.csv', index=False)