import pandas as pd
import numpy as np

dependent = pd.read_csv('../data/model/dependent/dependent_n.csv')
orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
aisles = pd.read_csv('../data/driver/driver_product.csv')[['product_id','aisle_id']]
driver = pd.read_csv('../data/profile/user_product_profile.csv')
driver = driver[['user_id','product_id','usr_ais_cnt']]
dependent = dependent[['user_id','product_id']]
target = orders[orders['counter'] == 1]
orders = orders[orders['counter'] > 1]
history = history.merge(orders, on='order_id', how='inner')
target = dependent.merge(target, on=['user_id'], how='inner')

target = target.merge(aisles, on='product_id')
target = target.merge(driver, on=['user_id','product_id'])
features = ['user_id','product_id','aisle_id','order_dow','order_hour_of_day']
features += ['days_since_prior_order','order_number','usr_ais_cnt']
target = target[features]
print(target.shape)

aggregate = {'order_id':'count'}

usr_ais_dow = history.groupby(['user_id','aisle_id','order_dow']).agg(aggregate).reset_index()
usr_ais_dow = usr_ais_dow.rename(columns={'order_id':'usr_ais_dow_cnt'})

history['order_hour_of_day'], hour = pd.qcut(history['order_hour_of_day'], 10, retbins=True, labels=False)
usr_ais_hod = history.groupby(['user_id','aisle_id','order_hour_of_day']).agg(aggregate).reset_index()
usr_ais_hod = usr_ais_hod.rename(columns={'order_id':'usr_ais_hod_cnt'})

history['days_since_prior_order'], days = pd.qcut(history['days_since_prior_order'], 7, retbins=True, labels=False)
usr_ais_dysc = history.groupby(['user_id','aisle_id','days_since_prior_order']).agg(aggregate).reset_index()
usr_ais_dysc = usr_ais_dysc.rename(columns={'order_id':'usr_ais_dysc_cnt'})

target['order_hour_of_day'] = pd.cut(target['order_hour_of_day'], bins=hour, labels=False, include_lowest=True)
target['days_since_prior_order'] = pd.cut(target['days_since_prior_order'], bins=days, labels=False, include_lowest=True)

del history
del orders
del dependent

target = target.merge(usr_ais_dow, on=['user_id','aisle_id','order_dow'], how='left')
target = target.merge(usr_ais_hod, on=['user_id','aisle_id','order_hour_of_day'], how='left')
target = target.merge(usr_ais_dysc, on=['user_id','aisle_id','days_since_prior_order'], how='left')
target = target.drop(['order_dow','order_hour_of_day','days_since_prior_order','order_number'], axis=1)
target['usrais2way1'] = target['usr_ais_dow_cnt'] / target['usr_ais_cnt'] 
target['usrais2way2'] = target['usr_ais_hod_cnt'] / target['usr_ais_cnt'] 
target['usrais2way3'] = target['usr_ais_dysc_cnt'] / target['usr_ais_cnt']
target = target.drop(['aisle_id','usr_ais_cnt'], axis=1)
target.to_csv('../data/profile/user_aisle_2way.csv', index=False)
