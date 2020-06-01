import pandas as pd
import numpy as np

def holdout(user_id, eval_set):
    if eval_set == 'train':
        if user_id % 10 >= 8:
            return 'valid'
        else:
            return eval_set
    else:
        return eval_set
    
users = pd.read_csv('../data/download/orders.csv')[['user_id','eval_set']]
users = users[users['eval_set'] != 'prior']
users = users.drop_duplicates()
users['eval_set'] = users.apply(lambda x : holdout(x['user_id'], x['eval_set']), axis=1)
print(users['eval_set'].value_counts())
users.to_csv('../data/driver/driver_user.csv', index=False)

orders = pd.read_csv('../data/download/orders.csv')
orders = orders.sort_values(['user_id', 'order_number'], ascending = [True, False])
orders['counter'] = orders.groupby('user_id')['order_number'].rank(ascending=False)
orders = orders.sort_values(by=['user_id','counter'], ascending=[True,True])
print(orders.shape)
orders.to_csv('../data/driver/driver_order.csv', index=False)

products = pd.read_csv('../data/download/products.csv').drop('product_name',axis=1)
products = products.append(pd.DataFrame([[0,-1,-1]], columns=products.columns))
products = products.sort_values(by=['product_id'])
print(products.shape)
products.to_csv('../data/driver/driver_product.csv', index=False)

order_prior = pd.read_csv('../data/download/order_products__prior.csv')
order_train = pd.read_csv('../data/download/order_products__train.csv')
order_prods = order_prior.append(order_train)
print(order_prods.shape)

none_candidate = pd.read_csv('../data/download/orders.csv')
none_candidate = none_candidate[none_candidate['order_number'] > 1][['order_id']]
none_orders = order_prods.groupby('order_id')['reordered'].max().reset_index()
none_orders = none_orders[none_orders['reordered'] == 0][['order_id']]
none_orders = none_orders.merge(none_candidate, on='order_id', how='inner')
none_orders['product_id'] = [0] * none_orders.shape[0]
none_orders['add_to_cart_order'] = [0] * none_orders.shape[0]
none_orders['reordered'] = [1] * none_orders.shape[0]
print(none_candidate.shape, none_orders.shape)

order_prods = order_prods.append(none_orders).sort_values(by=['order_id', 'add_to_cart_order'])
order_prods = order_prods.reset_index(drop=True)
order_prods = order_prods.merge(products, on=['product_id'], how = 'inner')
columns = ['order_id', 'product_id', 'aisle_id', 'department_id']
columns += ['add_to_cart_order','reordered']
order_prods = order_prods[columns]
order_prods = order_prods.sort_values(by=['order_id','add_to_cart_order'])
print(order_prods.shape)
order_prods.to_csv('../data/driver/driver_order_products.csv', index=False)