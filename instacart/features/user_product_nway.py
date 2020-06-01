import pandas as pd
import numpy as np

dependent = pd.read_csv('../data/model/dependent/dependent_n.csv')
dependent = dependent[['user_id','product_id']]
orders = pd.read_csv('../data/driver/driver_order.csv')
target = orders[orders['counter'] == 1]
orders = orders[orders['counter'] > 1]
products = pd.read_csv('../data/profile/user_product_profile.csv')
products = products[['user_id','product_id','usr_prd_cnt']]
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
target = dependent.merge(target, on=['user_id'], how='inner')

target = target[['user_id','product_id','order_dow','order_hour_of_day','days_since_prior_order','order_number']]
target = target.merge(products, on=['user_id','product_id'])
print(target.shape)

aggregate = {'order_id':'count'}

usr_prd_dow = history.groupby(['user_id','product_id','order_dow']).agg(aggregate).reset_index()
usr_prd_dow = usr_prd_dow.rename(columns={'order_id':'usr_prd_dow_cnt'})

history['order_hour_of_day'], hour = pd.qcut(history['order_hour_of_day'], 10, retbins=True, labels=False)
usr_prd_hod = history.groupby(['user_id','product_id','order_hour_of_day']).agg(aggregate).reset_index()
usr_prd_hod = usr_prd_hod.rename(columns={'order_id':'usr_prd_hod_cnt'})

history['days_since_prior_order'], days = pd.qcut(history['days_since_prior_order'], 7, retbins=True, labels=False)
usr_prd_dysc = history.groupby(['user_id','product_id','days_since_prior_order']).agg(aggregate).reset_index()
usr_prd_dysc = usr_prd_dysc.rename(columns={'order_id':'usr_prd_dysc_cnt'})

target['order_hour_of_day'] = pd.cut(target['order_hour_of_day'], bins=hour, labels=False, include_lowest=True)
target['days_since_prior_order'] = pd.cut(target['days_since_prior_order'], bins=days, labels=False, include_lowest=True)

del history
del orders
del dependent

target = target.merge(usr_prd_dow, on=['user_id','product_id','order_dow'], how='left')
target = target.merge(usr_prd_hod, on=['user_id','product_id','order_hour_of_day'], how='left')
target = target.merge(usr_prd_dysc, on=['user_id','product_id','days_since_prior_order'], how='left')
target = target.drop(['order_dow','order_hour_of_day','days_since_prior_order','order_number'], axis=1)

target['usrprd2way1'] = target['usr_prd_dow_cnt'] / target['usr_prd_cnt'] 
target['usrprd2way2'] = target['usr_prd_hod_cnt'] / target['usr_prd_cnt'] 
target['usrprd2way3'] = target['usr_prd_dysc_cnt'] / target['usr_prd_cnt']
target = target.drop(['usr_prd_cnt'], axis=1)
target.to_csv('../data/profile/user_product_2way.csv', index=False)