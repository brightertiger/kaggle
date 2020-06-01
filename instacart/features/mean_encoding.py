import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

users = pd.read_csv('../data/driver/driver_user.csv')
orders = pd.read_csv('../data/driver/driver_order.csv').drop('eval_set',axis=1)
products = pd.read_csv('../data/driver/driver_order_products.csv')
print(products.shape)
products = products.merge(orders, on='order_id', how='inner')
products = products.merge(users, on='user_id', how='inner')
print(products.shape)

def barreca(posterior, n,k,f, prior=0.10):
    factor = np.exp((n-k)/f)
    factor = factor / (factor + 1)
    if np.isnan(factor):
        factor = 1.
    return factor * posterior + (1 - factor) * prior

def target(cutoff):
    candidate = products[products['counter'] > cutoff][['user_id','product_id','aisle_id','eval_set']]
    nones = candidate[['user_id','eval_set']].drop_duplicates()
    nones['product_id'] = [0] * nones.shape[0]
    candidate = candidate.append(nones).drop_duplicates()
    dependent = products[products['counter'] == cutoff][['user_id','product_id','reordered']]
    user_list = list(set(dependent['user_id']))
    data = candidate.merge(dependent, on=['user_id','product_id'], how='left', indicator=True)
    data = data[data['user_id'].isin(user_list)]
    data['reordered'] = data['reordered'].fillna(0)
    data = data[['user_id','product_id','aisle_id','eval_set','reordered']]
    return data

def cumulative(counters, groupby):
    for counter in counters:
        if counter % 5 == 0:
            print(counter, 'status completed...')
        status = target(counter)
        reorder = status.groupby(groupby)['reordered'].sum().reset_index()
        potential = status.groupby(groupby)['reordered'].count().reset_index()
        reorder = reorder.rename(columns={'reordered':'sum'})
        potential = potential.rename(columns={'reordered':'count'})
        grouped = reorder.merge(potential, on=groupby, how='inner')
        grouped['counter'] = [counter] * grouped.shape[0]
        global cumulative_df
        cumulative_df = cumulative_df.append(grouped)
    return None

cumulative_df = pd.DataFrame()
cumulative(range(2,30),'aisle_id')
aggregate = {}
aggregate['sum'] = 'sum'
aggregate['count'] = 'sum'
aisle = cumulative_df.groupby(['aisle_id']).agg(aggregate).reset_index()
aisle['ais_post'] = aisle['sum'] / aisle['count']
aisle = aisle.drop(['sum','count'],axis=1)

cumulative_df = pd.DataFrame()
cumulative(range(2,30),['product_id','aisle_id'])
aggregate = {}
aggregate['sum'] = 'sum'
aggregate['count'] = 'sum'
product = cumulative_df.groupby(['product_id','aisle_id']).agg(aggregate).reset_index()
product['prd_post'] = product['sum'] / product['count']
product = product.drop(['sum','count'],axis=1)
product = product.merge(aisle, on='aisle_id', how='inner')
product['prd_ais_post_rt'] = product['prd_post'] / product['ais_post']
product = product.drop(['aisle_id'], axis=1)

cumulative_df = pd.DataFrame()
cumulative(range(2,30),'user_id')
aggregate = {}
aggregate['sum'] = 'sum'
aggregate['count'] = 'sum'
user = cumulative_df.groupby('user_id').agg(aggregate).reset_index()
user['usr_post'] = user['sum'] / user['count']
user = user.drop(['sum','count'],axis=1)

product.to_csv('../data/profile/product_brrc_profile.csv', index=False)
user.to_csv('../data/profile/user_brrc_profile.csv', index=False)