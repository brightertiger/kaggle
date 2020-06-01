import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

orders = pd.read_csv('../data/driver/driver_order.csv')
orders = orders[orders['counter'] > 1]
products = pd.read_csv('../data/driver/driver_order_products.csv')
print(products.shape)
products = products.merge(orders, on='order_id', how='inner')
products = products.drop(['eval_set'], axis=1)
print(products.shape)

aggregate = {}
aggregate['reordered'] = np.sum
aggregate['counter'] = np.count_nonzero
aggregate['order_id'] = pd.Series.nunique
aggregate['product_id'] = pd.Series.nunique
aggregate['aisle_id'] = pd.Series.nunique
aggregate['department_id'] = pd.Series.nunique
aggregate['days_since_prior_order'] = np.median
user_common = products.groupby('user_id').agg(aggregate).reset_index()
features  = ['usr_sum_rdr', 'usr_cnt_prd', 'usr_cnt_ord','usr_cds_prd']
features +=['usr_cds_ais','usr_cds_dep','usr_med_dysc']
user_common.columns = ['user_id'] + features

cart_length = products.groupby(['user_id','order_id'])['product_id'].count().reset_index()
cart_length = cart_length.groupby(['user_id'])['product_id'].mean().reset_index()
cart_length = cart_length.rename(columns={'product_id':'cartlen'})

cart_diverse = products.groupby(['user_id','order_id'])['aisle_id'].apply(pd.Series.nunique).reset_index()
cart_diverse = cart_diverse.groupby(['user_id'])['aisle_id'].mean().reset_index()
cart_diverse = cart_diverse.rename(columns={'aisle_id':'cartdiv'})

avg_reorder = products.groupby(['user_id','order_id'])['reordered'].mean().reset_index()
avg_reorder = avg_reorder.groupby(['user_id'])['reordered'].mean().reset_index()
avg_reorder = avg_reorder.rename(columns={'reordered':'usr_avg_rdr'})

lag_reorder = pd.read_csv('../data/model/dependent/dependent_n_1.csv')
lag_reorder = lag_reorder.groupby('user_id')['reordered'].mean().reset_index()
lag_reorder = lag_reorder.rename(columns={'reordered':'usr_lag_rdr'})

user_profile = pd.read_csv('../data/driver/driver_user.csv')
user_profile = user_profile.merge(user_common, on='user_id',  how='left')
user_profile = user_profile.merge(cart_length, on='user_id', how='left')
user_profile = user_profile.merge(cart_diverse, on='user_id', how='left')
user_profile = user_profile.merge(avg_reorder, on='user_id', how='left')
user_profile = user_profile.merge(lag_reorder, on='user_id', how='left')

user_profile.to_csv('../data/profile/user_profile.csv', index=False)