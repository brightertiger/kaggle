import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

dependent = pd.read_csv('../data/model/dependent/dependent_n.csv')
orders = pd.read_csv('../data/driver/driver_order.csv')
products = pd.read_csv('../data/profile/product_basic_profile.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
aisles = pd.read_csv('../data/driver/driver_product.csv')[['product_id','aisle_id']]
dependent = dependent[['user_id','product_id']]
target = orders[orders['counter'] == 1]
orders = orders[orders['counter'] > 1]
products = products[['product_id','ais_sum_rdr']]
history = history.merge(orders, on='order_id', how='inner')
target = dependent.merge(target, on=['user_id'], how='inner')

target = target.merge(aisles, on='product_id')
target = target.merge(products, on='product_id')
features = ['user_id','product_id','aisle_id','order_dow','order_hour_of_day']
features += ['days_since_prior_order','order_number','ais_sum_rdr']
target = target[features]
print(target.shape)

aggregate = {'order_id':'count'}

ais_dow = history.groupby(['aisle_id','order_dow']).agg(aggregate).reset_index()
ais_dow = ais_dow.rename(columns={'order_id':'ais_dow_cnt'})

history['order_hour_of_day'], hour = pd.qcut(history['order_hour_of_day'], 10, retbins=True, labels=False)
ais_hod = history.groupby(['aisle_id','order_hour_of_day']).agg(aggregate).reset_index()
ais_hod = ais_hod.rename(columns={'order_id':'ais_hod_cnt'})

history['days_since_prior_order'], days = pd.qcut(history['days_since_prior_order'], 5, retbins=True, labels=False)
ais_dysc = history.groupby(['aisle_id','days_since_prior_order']).agg(aggregate).reset_index()
ais_dysc = ais_dysc.rename(columns={'order_id':'ais_dysc_cnt'})

history['order_number'], orders = pd.qcut(history['order_number'], 10, retbins=True, labels=False)
ais_ordn = history.groupby(['aisle_id','order_number']).agg(aggregate).reset_index()
ais_ordn = ais_ordn.rename(columns={'order_id':'ais_ordn_cnt'})

target['order_hour_of_day'] = pd.cut(target['order_hour_of_day'], bins=hour, labels=False, include_lowest=True)
target['days_since_prior_order'] = pd.cut(target['days_since_prior_order'], bins=days, labels=False, include_lowest=True)
target['order_number'] = pd.cut(target['order_number'], bins=orders, labels=False, include_lowest=True)

target = target.merge(ais_dow, on=['aisle_id','order_dow'], how='left')
target = target.merge(ais_hod, on=['aisle_id','order_hour_of_day'], how='left')
target = target.merge(ais_dysc, on=['aisle_id','days_since_prior_order'], how='left')
target = target.merge(ais_ordn, on=['aisle_id','order_number'], how='left')
target = target.drop(['aisle_id','order_dow','order_hour_of_day','days_since_prior_order','order_number'], axis=1)

target['ais2way1'] = target['ais_dow_cnt'] / (target['ais_sum_rdr'] + 1)
target['ais2way2'] = target['ais_hod_cnt'] / (target['ais_sum_rdr'] + 1)
target['ais2way3'] = target['ais_dysc_cnt'] / (target['ais_sum_rdr'] + 1)
target['ais2way4'] = target['ais_ordn_cnt'] / (target['ais_sum_rdr'] + 1)
target = target.drop(['ais_sum_rdr'], axis=1)
target.to_csv('../data/profile/aisle_2way.csv', index=False)