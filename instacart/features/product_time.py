import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

orders = pd.read_csv('../data/driver/driver_order.csv')
products = pd.read_csv('../data/driver/driver_order_products.csv')
orders = orders[['user_id','order_id','counter','days_since_prior_order']]
orders['days'] = orders.groupby('user_id')['days_since_prior_order'].shift(1)
orders = orders[orders['counter'] > 1].drop('days_since_prior_order',axis=1)
orders['cum_days'] = orders.groupby('user_id')['days'].cumsum()
orders = orders.drop('days',axis=1)
data = products.merge(orders, on=['order_id'])

product_time = data[['user_id','product_id','cum_days','reordered']]
product_time = product_time.sort_values(by=['user_id','product_id','cum_days'], ascending=[True,True,False])
product_time['shift_days'] = product_time.groupby(['user_id','product_id'])['cum_days'].shift(1)
product_time = product_time[np.logical_not(product_time['shift_days'].isnull())]
product_time['days_diff'] = product_time['shift_days'] - product_time['cum_days']
product_time = product_time.groupby('product_id')['days_diff'].apply(np.median).reset_index()
product_time.columns = ['product_id','prd_med_dydiff']
product_time['prd_med_dydiff'].describe()

product_order = data[['user_id','product_id','counter']]
product_order = product_order.sort_values(by=['user_id','product_id','counter'])
product_order['shift_counter'] = product_order.groupby(['user_id','product_id'])['counter'].shift(1)
product_order = product_order[np.logical_not(product_order['shift_counter'].isnull())]
product_order['counter_diff'] = product_order['counter'] - product_order['shift_counter']
product_order = product_order.groupby('product_id')['counter_diff'].apply(np.nanmedian).reset_index()
product_order.columns = ['product_id','prd_med_orddiff']
product_order['prd_med_orddiff'].describe()

profile = pd.read_csv('../data/driver/driver_product.csv')
profile = profile.merge(product_time, on='product_id',how='left')
profile = profile.merge(product_order, on='product_id',how='left')
profile = profile.drop(['department_id','aisle_id'],axis=1).fillna(0.)
profile.to_csv('../data/profile/product_time_profile.csv', index=False)