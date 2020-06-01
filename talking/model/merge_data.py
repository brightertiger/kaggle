import pandas as pd
import numpy as np

ip = pd.read_feather('../data/data/features/count/ip.feather').astype('uint32')
app = pd.read_feather('../data/data/features/count/app.feather').astype('uint32')
os = pd.read_feather('../data/data/features/count/os.feather').astype('uint32')

ip_day_hour = pd.read_feather('../data/data/features/count/ip_day_hour.feather').astype('uint32')
ip_app = pd.read_feather('../data/data/features/count/ip_app.feather').astype('uint32')
ip_app_os = pd.read_feather('../data/data/features/count/ip_app_os.feather').astype('uint32')
ip_device = pd.read_feather('../data/data/features/count/ip_device.feather').astype('uint32')
app_channel = pd.read_feather('../data/data/features/count/app_channel.feather').astype('uint32')
ip_hour_os = pd.read_feather('../data/data/features/count/ip_hour_os.feather').astype('uint32')
ip_hour_app = pd.read_feather('../data/data/features/count/ip_hour_app.feather').astype('uint32')

user = pd.read_feather('../data/data/features/count/user_count.feather').astype('uint32')
user_app = pd.read_feather('../data/data/features/count/user_app_count.feather').astype('uint32')

ip_app_unq = pd.read_feather('../data/data/features/unique/ip_app_unq.feather').astype('uint32')
ip_channel_unq = pd.read_feather('../data/data/features/unique/ip_channel_unq.feather').astype('uint32')

next_click_train = pd.read_feather('../data/data/features/next_click/next_click_train.feather')
next_click_test = pd.read_feather('../data/data/features/next_click/next_click_test.feather')
next_click_train_1 = pd.read_feather('../data/data/features/next_click/next_click_train_1.feather')
next_click_test_1 = pd.read_feather('../data/data/features/next_click/next_click_test_1.feather')

ip_enc_train = pd.read_feather('../data/data/features/running/train_ip.feather')
ip_enc_valid = pd.read_feather('../data/data/features/running/valid_ip.feather')
ip_enc_test = pd.read_feather('../data/data/features/running/test_ip.feather')

ip_app_enc_train = pd.read_feather('../data/data/features/running/train_ip_app.feather')
ip_app_enc_valid = pd.read_feather('../data/data/features/running/valid_ip_app.feather')
ip_app_enc_test = pd.read_feather('../data/data/features/running/test_ip_app.feather')

ip_rank = pd.read_feather('../data/data/features/rank/ip.feather')
app_channel_rank = pd.read_feather('../data/data/features/rank/app_channel.feather')
app_os_rank = pd.read_feather('../data/data/features/rank/app_os.feather')
channel_os_rank = pd.read_feather('../data/data/features/rank/channel_os.feather')

def reduce_size(data):
    for column in list(data.columns):
        if data[column].max() <= 250:
            data[column] = data[column].astype('uint8')
        elif data[column].max() <= 65000 and data[column].min() >= 0:
            data[column] = data[column].astype('uint16')
    return data
    
def merge(data, mode='train'):
    print('merge starts..', data.shape)
    # one way counts
    data = data.merge(ip, on='ip')
    data = data.merge(os, on='os')
    data = data.merge(app, on='app')
    data = reduce_size(data)
    print('one way merge complete...', data.shape)
    # n way counts
    data = data.merge(ip_day_hour, on=['ip','day','hour'])
    data = data.merge(ip_app, on=['ip','app'])
    data = data.merge(ip_app_os, on=['ip','app','os'])
    data = data.merge(ip_device, on=['ip','device'])
    data = data.merge(app_channel, on=['app','channel'])
    data = data.merge(ip_hour_os, on=['ip','hour','os'])
    data = data.merge(ip_hour_app, on=['ip','hour','app'])
    data = reduce_size(data)
    print('n way merge complete...', data.shape)
    # unique counts
    data = data.merge(ip_app_unq, on=['ip'])
    data = data.merge(ip_channel_unq, on=['ip'])
    data = reduce_size(data)
    # user counts
    data = data.merge(user, on=['ip','device','os'])
    data = data.merge(user_app, on=['ip','device','os','app'])
    data = reduce_size(data)
    print('user counts merge complete...', data.shape)
    # rank features
    data = data.merge(ip_rank, on=['ip'], how='left')
    data = data.merge(app_channel_rank, on=['app','channel'], how='left')
    data = data.merge(app_os_rank, on=['app','os'], how='left')
    data = data.merge(channel_os_rank, on=['channel','os'], how='left')
    data = data.fillna(11)
    print('rank features merge complete...', data.shape)
    # other features
    if mode == 'train':
        data = data.merge(next_click_train, on='click_id')
        data = data.merge(ip_enc_train, on='click_id', how='left')
        data = data.merge(ip_app_enc_train, on='click_id', how='left')
        data = data.fillna(0)
        data = reduce_size(data)
        print('other merge complete...', data.shape)
    elif mode == 'valid':
        data = data.merge(next_click_train, on='click_id')
        data = data.merge(ip_enc_valid, on='ip', how='left')
        data = data.merge(ip_app_enc_valid, on=['ip','app'], how='left')
        data = data.fillna(0)
        data = reduce_size(data)
        print('other merge complete...', data.shape)
    else:
        data = data.merge(next_click_test, on='click_id', how='left')
        data = data.merge(ip_enc_test, on='ip', how='left')
        data = data.merge(ip_app_enc_test, on=['ip','app'], how='left')
        data = data.fillna(0)
        data = reduce_size(data)
        print('other merge complete...', data.shape)
    data = data.reset_index(drop=True)
    return data

train_data = pd.read_feather('../data/data/files/train_data.feather')
train_data = merge(train_data)
train_data.to_feather('../data/data/model/train_data.feather')
del train_data

valid_data = pd.read_feather('../data/data/files/valid_data.feather')
valid_data = merge(valid_data, 'valid')
valid_data.to_feather('../data/data/model/valid_data.feather')
del valid_data

score_data = pd.read_feather('../data/data/files/score_data.feather')
score_data = merge(score_data,'score')
score_data.to_feather('../data/data/model/score_data.feather')
del score_data