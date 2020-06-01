import pandas as pd
import numpy as np

users = pd.read_csv('../data/driver/driver_user.csv')
orders = pd.read_csv('../data/driver/driver_order.csv').drop('eval_set',axis=1)
products = pd.read_csv('../data/driver/driver_order_products.csv')
print(products.shape)
products = products.merge(orders, on='order_id', how='inner')
products = products.merge(users, on='user_id', how='inner')
print(products.shape)

def target(cutoff):
    candidate = products[products['counter'] > cutoff][['user_id','product_id','eval_set']]
    nones = users[['user_id','eval_set']]
    nones['product_id'] = [0] * users.shape[0]
    candidate = candidate.append(nones).drop_duplicates()
    dependent = products[products['counter'] == cutoff][['user_id','product_id','reordered']]
    data = candidate.merge(dependent, on=['user_id','product_id'], how='left', indicator=True)
    data['reordered'] = data['reordered'].fillna(0)
    print('reorder rate', data[data['eval_set'] != 'test']['reordered'].mean())
    data = data[['user_id','product_id','eval_set','reordered']]
    return data

dependent_n = target(1)
dependent_n.to_csv('../data/model/dependent/dependent_n.csv', index=False)

dependent_n_1 = target(2)
dependent_n_1.to_csv('../data/model/dependent/dependent_n_1.csv', index=False)

dependent_n_2 = target(3)
dependent_n_2.to_csv('../data/model/dependent/dependent_n_2.csv', index=False)