import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

dependent = pd.read_csv('../data/model/dependent/dependent_n.csv')
dependent = dependent[['user_id','product_id']]
orders = pd.read_csv('../data/driver/driver_order.csv')
target = orders[orders['counter'] == 1]
orders = orders[orders['counter'] > 1]
users = pd.read_csv('../data/profile/user_profile.csv')
users = users[['user_id','usr_cnt_ord']]
target = dependent.merge(target, on=['user_id'], how='inner')

target = target[['user_id','product_id','order_dow','order_hour_of_day','days_since_prior_order','order_number']]
target = target.merge(users, on='user_id')
print(target.shape)

aggregate = {'order_id': 'count'}

usr_dow = orders.groupby(['user_id','order_dow']).agg(aggregate).reset_index()
usr_dow = usr_dow.rename(columns={'order_id':'usr_dow_cnt'})

orders['order_hour_of_day'], hour = pd.qcut(orders['order_hour_of_day'], 10, retbins=True, labels=False)
usr_hod = orders.groupby(['user_id','order_hour_of_day']).agg(aggregate).reset_index()
usr_hod = usr_hod.rename(columns={'order_id':'usr_hod_cnt'})

orders['days_since_prior_order'], days = pd.qcut(orders['days_since_prior_order'], 5, retbins=True, labels=False)
usr_dysc = orders.groupby(['user_id','days_since_prior_order']).agg(aggregate).reset_index()
usr_dysc = usr_dysc.rename(columns={'order_id':'usr_dysc_cnt'})

target['order_hour_of_day'] = pd.cut(target['order_hour_of_day'], bins=hour, labels=False, include_lowest=True)
target['days_since_prior_order'] = pd.cut(target['days_since_prior_order'], bins=days, labels=False, include_lowest=True)

target = target.merge(usr_dow, on=['user_id','order_dow'], how='left')
target = target.merge(usr_hod, on=['user_id','order_hour_of_day'], how='left')
target = target.merge(usr_dysc, on=['user_id','days_since_prior_order'], how='left')
target = target.drop(['order_dow','order_hour_of_day','days_since_prior_order','order_number'], axis=1)
target = target.drop(['usr_cnt_ord'], axis=1)
target.to_csv('../data/profile/user_2way.csv', index=False)