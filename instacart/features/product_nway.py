import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

dependent = pd.read_csv('../data/model/dependent/dependent_n.csv')
dependent = dependent[['user_id','product_id']]
orders = pd.read_csv('../data/driver/driver_order.csv')
target = orders[orders['counter'] == 1]
orders = orders[orders['counter'] > 1]
products = pd.read_csv('../data/profile/product_basic_profile.csv')
products = products[['product_id','prd_sum_rdr']]
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
target = dependent.merge(target, on=['user_id'], how='inner')

target = target[['user_id','product_id','order_dow','order_hour_of_day','days_since_prior_order','order_number']]
target = target.merge(products, on='product_id')
print(target.shape)

aggregate = {'order_id':'count'}

prd_dow = history.groupby(['product_id','order_dow']).agg(aggregate).reset_index()
prd_dow = prd_dow.rename(columns={'order_id':'prd_dow_cnt'})

history['order_hour_of_day'], hour = pd.qcut(history['order_hour_of_day'], 10, retbins=True, labels=False)
prd_hod = history.groupby(['product_id','order_hour_of_day']).agg(aggregate).reset_index()
prd_hod = prd_hod.rename(columns={'order_id':'prd_hod_cnt'})

history['days_since_prior_order'], days = pd.qcut(history['days_since_prior_order'], 5, retbins=True, labels=False)
prd_dysc = history.groupby(['product_id','days_since_prior_order']).agg(aggregate).reset_index()
prd_dysc = prd_dysc.rename(columns={'order_id':'prd_dysc_cnt'})

history['order_number'], orders = pd.qcut(history['order_number'], 10, retbins=True, labels=False)
prd_ordn = history.groupby(['product_id','order_number']).agg(aggregate).reset_index()
prd_ordn = prd_ordn.rename(columns={'order_id':'prd_ordn_cnt'})

target['order_hour_of_day'] = pd.cut(target['order_hour_of_day'], bins=hour, labels=False, include_lowest=True)
target['days_since_prior_order'] = pd.cut(target['days_since_prior_order'], bins=days, labels=False, include_lowest=True)
target['order_number'] = pd.cut(target['order_number'], bins=orders, labels=False, include_lowest=True)

target = target.merge(prd_dow, on=['product_id','order_dow'], how='left')
target = target.merge(prd_hod, on=['product_id','order_hour_of_day'], how='left')
target = target.merge(prd_dysc, on=['product_id','days_since_prior_order'], how='left')
target = target.merge(prd_ordn, on=['product_id','order_number'], how='left')
target = target.drop(['order_dow','order_hour_of_day','days_since_prior_order','order_number'], axis=1)
target['prd2way1'] = target['prd_dow_cnt'] / (target['prd_sum_rdr'] + 1)
target['prd2way2'] = target['prd_hod_cnt'] / (target['prd_sum_rdr'] + 1)
target['prd2way3'] = target['prd_dysc_cnt'] / (target['prd_sum_rdr'] + 1)
target['prd2way4'] = target['prd_ordn_cnt'] / (target['prd_sum_rdr'] + 1)
target = target.drop(['prd_sum_rdr'], axis=1)
target.to_csv('../data/profile/product_2way.csv', index=False)