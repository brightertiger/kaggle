import pandas as pd
import numpy as np

train_driver = pd.read_csv('../data/download/train.csv', usecols=['item_id'])
test_driver = pd.read_csv('../data/download/test.csv', usecols=['item_id'])
print('data:', train_driver.shape, test_driver.shape)

feats = ['title_p','desc_p','title_trainable_p','desc_trainable_p','title_w2v_p','desc_w2v_p']
train_rnn = pd.read_hdf('../data/data/features/moiz/data_all_train.h5')[['item_id'] +feats]
test_rnn = pd.read_hdf('../data/data/features/moiz/data_all_test.h5')[['item_id'] +feats]
train_rnn.columns = ['item_id','rnn_1','rnn_2','rnn_3','rnn_4','rnn_5','rnn_6']
test_rnn.columns = ['item_id','rnn_1','rnn_2','rnn_3','rnn_4','rnn_5','rnn_6']

train_driver = train_driver.merge(train_rnn, on='item_id')
test_driver = test_driver.merge(test_rnn, on='item_id')
print('data:', train_driver.shape, test_driver.shape)

def merge(train, test, name):
    global train_driver, test_driver
    train_feat = pd.read_csv(train)
    test_feat = pd.read_csv(test)
    train_feat = train_feat.rename(columns={'deal_probability':name})
    test_feat = test_feat.rename(columns={'deal_probability':name})
    train_driver = train_driver.merge(train_feat, on='item_id', how='left')
    test_driver = test_driver.merge(test_feat, on='item_id',  how='left')
    train_driver = train_driver.fillna(0.)
    test_driver = test_driver.fillna(0.)
    return None

path = '../data/data/features/image/'
merge(path + 'vgg_train.csv', path + 'vgg_test.csv','vgg_1')
merge(path + 'vgg_train_1.csv', path + 'vgg_test_1.csv','vgg_2')
merge(path + 'xception_train.csv', path + 'xception_test.csv','xcept')
print('data:', train_driver.shape, test_driver.shape)

ridge_1 = pd.read_csv('../data/data/features/ridge/ridge_1.csv')
ridge_2 = pd.read_csv('../data/data/features/ridge/ridge_2.csv')
train_driver = train_driver.merge(ridge_1, on='item_id')
test_driver = test_driver.merge(ridge_1, on='item_id')
train_driver = train_driver.merge(ridge_2, on='item_id')
test_driver = test_driver.merge(ridge_2, on='item_id')
print('data:', train_driver.shape, test_driver.shape)

train_driver.to_csv('../data/data/features/moiz/train_weak.csv', index=False)
test_driver.to_csv('../data/data/features/moiz/test_weak.csv', index=False)