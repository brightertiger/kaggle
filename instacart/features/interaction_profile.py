import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

orders = pd.read_csv('../data/driver/driver_order.csv')
products = pd.read_csv('../data/driver/driver_product.csv')
prior = pd.read_csv('../data/profile/product_brrc_profile.csv')
prior = prior[['product_id','prd_post']]
reorder_0 = pd.read_csv('../data/model/dependent/dependent_n.csv')
reorder_1 = pd.read_csv('../data/model/dependent/dependent_n_1.csv')
orders_0 = orders[orders['counter'] == 1].drop(['eval_set','order_id','counter'],axis=1)
orders_1 = orders[orders['counter'] == 2].drop(['eval_set','order_id','counter'],axis=1)
reorder_0 = reorder_0.merge(orders_0, on='user_id', how='inner')
reorder_1 = reorder_1.merge(orders_1, on='user_id', how='inner')
reorder_0 = reorder_0.merge(products, on='product_id', how='inner')
reorder_1 = reorder_1.merge(products, on='product_id', how='inner')
reorder_0 = reorder_0.merge(prior, on='product_id', how='inner')
reorder_1 = reorder_1.merge(prior, on='product_id', how='inner')

feats = ['order_dow','order_hour_of_day','days_since_prior_order','order_number']
target = reorder_0[['user_id','product_id','aisle_id','reordered'] + feats]

def barreca(posterior,prior,n,k=30,f=3):
    factor = np.exp(min(n-k,1000)/f)
    factor = factor / (factor + 1)
    if np.isnan(factor):
        factor = 1.
    return factor * posterior + (1 - factor) * prior

def aggregate_2way(data,level,prior, posterior):
    data = data.groupby(level + [prior]).agg({'reordered':['mean','count']})
    data = data.reset_index()
    data.columns = data.columns.droplevel(1)
    data.columns = level + [prior, posterior,'support']
    return data

prd_dow = aggregate_2way(reorder_1,['product_id','order_dow'],'prd_post','prd_dow_int')
prd_dow['prd_dow_brrc'] = prd_dow.apply(lambda x:barreca(x['prd_dow_int'],x['prd_post'], x['support']),axis=1)
prd_dow['prd_dow_int'] = prd_dow['prd_dow_brrc'] / prd_dow['prd_post']
prd_dow = prd_dow[['product_id','order_dow','prd_dow_int']]

reorder_1['order_hour_of_day'], hour = pd.qcut(reorder_1['order_hour_of_day'], 10, retbins=True, labels=False)
prd_hod = aggregate_2way(reorder_1,['product_id','order_hour_of_day'],'prd_post','prd_hod_int')
prd_hod['prd_hod_brrc'] = prd_hod.apply(lambda x:barreca(x['prd_hod_int'],x['prd_post'], x['support']),axis=1)
prd_hod['prd_hod_int'] = prd_hod['prd_hod_brrc'] / prd_hod['prd_post']
prd_hod = prd_hod[['product_id','order_hour_of_day','prd_hod_int','prd_hod_brrc']]

reorder_1['order_number'], orders = pd.qcut(reorder_1['order_number'], 10, retbins=True, labels=False)
prd_ordn = aggregate_2way(reorder_1,['product_id','order_number'],'prd_post','prd_ordn_int')
prd_ordn['prd_ordn_brrc'] = prd_ordn.apply(lambda x:barreca(x['prd_ordn_int'],x['prd_post'], x['support']),axis=1)
prd_ordn['prd_ordn_int'] = prd_ordn['prd_ordn_brrc'] / prd_ordn['prd_post']
prd_ordn = prd_ordn[['product_id','order_number','prd_ordn_int','prd_ordn_brrc']]

reorder_1['days_since_prior_order'], days = pd.qcut(reorder_1['days_since_prior_order'], 5, retbins=True, labels=False)
prd_dysc = aggregate_2way(reorder_1,['product_id','days_since_prior_order'],'prd_post','prd_dysc_int')
prd_dysc['prd_dysc_brrc'] = prd_dysc.apply(lambda x:barreca(x['prd_dysc_int'],x['prd_post'], x['support']),axis=1)
prd_dysc['prd_dysc_int'] = prd_dysc['prd_dysc_brrc'] / prd_dysc['prd_post']
prd_dysc = prd_dysc[['product_id','days_since_prior_order','prd_dysc_int','prd_dysc_brrc']]

reorder_0['order_hour_of_day'] = pd.cut(reorder_0['order_hour_of_day'], bins=hour, labels=False, include_lowest=True)
reorder_0['days_since_prior_order'] = pd.cut(reorder_0['days_since_prior_order'], bins=days, labels=False, include_lowest=True)
reorder_0['order_number'] = pd.cut(reorder_0['order_number'], bins=orders, labels=False, include_lowest=True)

target = reorder_0.merge(prd_dow, on=['product_id','order_dow'], how='left')
target = target.merge(prd_hod, on=['product_id','order_hour_of_day'], how='left')
target = target.merge(prd_ordn, on=['product_id','order_number'], how='left')
target = target.merge(prd_dysc, on=['product_id','days_since_prior_order'], how='left')
target = target[['user_id','product_id'] + list(target.columns[-4:])]
target.to_csv('../data/profile/product_interaction.csv', index=False)