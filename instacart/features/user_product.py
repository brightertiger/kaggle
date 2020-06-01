import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import warnings
warnings.simplefilter('ignore')

orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
history = history[history['counter'] > 1]
del orders

def aggregation(level, prefix):
    data =  history.groupby(level)['order_id'].count().reset_index()
    data.columns = level + [prefix + 'cnt']
    return data

user_total = history.groupby(['user_id'])['order_id'].count().reset_index()
user_total.columns = ['user_id','tot_cnt']
user_order = history.groupby(['user_id'])['order_id'].apply(pd.Series.nunique).reset_index()
user_order.columns = ['user_id','tot_ord']
user_prd = aggregation(['user_id','product_id','aisle_id','department_id'],'usr_prd_')
user_ais = aggregation(['user_id','aisle_id'],'usr_ais_')
user_dep = aggregation(['user_id','department_id'],'usr_dep_')
user_hist = user_prd.merge(user_ais, on=['user_id','aisle_id'], how='inner')
user_hist = user_hist.merge(user_dep, on=['user_id','department_id'], how='inner')
user_hist = user_hist.drop(['aisle_id','department_id'], axis=1)
user_hist = user_hist.merge(user_total, on=['user_id'], how='inner')
user_hist = user_hist.merge(user_order, on=['user_id'], how='inner')
user_hist['usr_prd_perc_cnt'] = user_hist['usr_prd_cnt'] / user_hist['tot_cnt']
user_hist['usr_ais_perc_cnt'] = user_hist['usr_ais_cnt'] / user_hist['tot_cnt']
user_hist['usr_prd_perc_ord'] = (user_hist['usr_prd_cnt']- 1) / (user_hist['tot_ord'] - 1)
user_hist['usr_ais_perc_ord'] = (user_hist['usr_ais_cnt']- 1) / (user_hist['tot_ord'] - 1)
user_hist = user_hist.drop(['tot_cnt','tot_ord'], axis=1)

reorder_rate = history.groupby(['user_id','product_id']).agg({'reordered':'sum','order_id':'count'})
reorder_rate = reorder_rate.reset_index()
reorder_rate.columns = ['user_id','product_id','reorders','count']
reorder_rate['usr_prd_rd_rt'] = reorder_rate['reorders'] / reorder_rate['count'] 
reorder_rate = reorder_rate.drop('count', axis=1)

def aggregation(level, prefix):
    data =  history.groupby(level)['order_number'].max().reset_index()
    data.columns = level + [prefix + 'lst_ord']
    return data

user_order = history.groupby('user_id')['order_number'].max().reset_index()
user_order.columns = ['user_id','last_order']
user_prd = aggregation(['user_id','product_id','aisle_id','department_id'],'usr_prd_')
user_ais = aggregation(['user_id','aisle_id'],'usr_ais_')
user_dep = aggregation(['user_id','department_id'],'usr_dep_')
user_last = user_prd.merge(user_ais, on=['user_id','aisle_id'], how='inner')
user_last = user_last.merge(user_dep, on=['user_id','department_id'], how='inner')
user_last = user_last.drop(['aisle_id','department_id'], axis=1)
user_last = user_last.merge(user_order, on=['user_id'], how='inner')
user_last['usr_prd_lst_ord'] = user_last['last_order'] - user_last['usr_prd_lst_ord'] 
user_last['usr_ais_lst_ord'] = user_last['last_order'] - user_last['usr_ais_lst_ord']
user_last['usr_prd_lst_ratio'] = user_last['usr_prd_lst_ord'] / user_last['last_order']
user_last = user_last.drop(['last_order','usr_dep_lst_ord'],axis=1)

orders = pd.read_csv('../data/driver/driver_order.csv')
orders = orders.sort_values(by=['user_id','counter'])
orders['days_shift'] = orders.groupby(['user_id'])['days_since_prior_order'].shift(1)
orders = orders[orders['counter'] > 1]
orders['cum_days'] = orders.groupby(['user_id'])['days_shift'].cumsum()
orders = orders[['order_id','user_id','cum_days']]
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
history = history[['user_id','product_id','aisle_id','cum_days']]
user_prd_days = history.groupby(['user_id','product_id','aisle_id'])['cum_days'].min().reset_index()
user_prd_days.columns = ['user_id','product_id','aisle_id','usr_prd_lstdy']
user_ais_days = history.groupby(['user_id','aisle_id'])['cum_days'].min().reset_index()
user_ais_days.columns = ['user_id','aisle_id','usr_ais_lstdy']
user_days = user_prd_days.merge(user_ais_days, on=['user_id','aisle_id'], how='inner')
user_days = user_days.drop(['aisle_id'],axis=1)

orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
history = history[history['counter'] > 1]
driver = history[['user_id','product_id']].drop_duplicates()
del orders
user_prd_frst = history.groupby(['user_id','product_id'])['order_number'].min().reset_index()
user_prd_frst = user_prd_frst.rename(columns={'order_number':'first_order'})
user_ord = history.groupby(['user_id'])['order_number'].max().reset_index()
user_ord = user_ord.rename(columns={'order_number':'usr_order'})
history = history[['user_id','product_id','order_number']]
history = history.merge(user_prd_frst, on=['user_id','product_id'], how='inner')
history = history.merge(user_ord, on=['user_id'], how='inner')
history = history[history['order_number'] > history['first_order']]
history = history.groupby(['user_id','product_id','usr_order','first_order'])['order_number'].count()
history = history.reset_index().rename(columns={'order_number':'ord_count'})
user_prd_frst_ord = history
user_prd_frst_ord['max_poss'] = user_prd_frst_ord['usr_order'] - user_prd_frst_ord['first_order']
user_prd_frst_ord['actual'] = user_prd_frst_ord['ord_count'] - 1
user_prd_frst_ord['usr_prd_fs_ord'] = user_prd_frst_ord['actual'] / user_prd_frst_ord['max_poss'] 
user_prd_frst_ord = user_prd_frst_ord[['user_id','product_id','usr_prd_fs_ord']]
user_prd_frst_ord = driver.merge(user_prd_frst_ord, on=['user_id','product_id'], how='left')
user_prd_frst_ord = user_prd_frst_ord.fillna(0)

orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
history = history[history['counter'] > 1]
del orders
add_to_cart = history.groupby(['user_id','product_id'])['add_to_cart_order'].agg(np.median)
add_to_cart = add_to_cart.reset_index().rename(columns={'add_to_cart_order':'usr_prd_adct'})
user_add_crt = history[['user_id','product_id']].drop_duplicates()
user_add_crt = user_add_crt.merge(add_to_cart, on=['user_id','product_id'], how='left')

history = pd.read_csv('../data/driver/driver_order_products.csv')
driver = pd.read_csv('../data/driver/driver_order.csv')
history = history.merge(driver, on='order_id', how='inner')
dependent = driver[driver['counter'] == 1][['user_id','order_hour_of_day']]
history = history[history['counter'] > 1]
last_order = history.groupby(['user_id','product_id'])['order_number'].max().reset_index()
last_order = history.merge(last_order, on=['user_id','product_id'], how='inner')
last_order = last_order[last_order['order_number_x'] == last_order['order_number_y']]
last_order = last_order[['user_id','product_id','order_hour_of_day']]
current_order = dependent[['user_id','order_hour_of_day']].drop_duplicates()
current_order = last_order.merge(current_order, on=['user_id'], how='inner')
current_order['diff_hod'] = current_order['order_hour_of_day_x'] - current_order['order_hour_of_day_y']
current_order['diff_hod'] = current_order['diff_hod'].map(lambda x : min(abs(x),24-abs(x)))
time_delta = current_order[['user_id','product_id','diff_hod']]

print(user_hist.shape, user_last.shape, user_days.shape)
print(user_prd_frst_ord.shape, time_delta.shape, user_add_crt.shape)
print(reorder_rate.shape)

user_hist = user_hist.merge(user_last, on=['user_id','product_id'], how='left')
user_hist = user_hist.merge(user_days, on=['user_id','product_id'], how='left')
user_hist = user_hist.merge(user_prd_frst_ord, on=['user_id','product_id'], how='left')
user_hist = user_hist.merge(time_delta, on=['user_id','product_id'], how='left')
user_hist = user_hist.merge(user_add_crt, on=['user_id','product_id'], how='left')
user_hist = user_hist.merge(reorder_rate, on=['user_id','product_id'], how='left')
user_hist.to_csv('../data/profile/user_product_profile.csv', index=False)
