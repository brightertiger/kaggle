import pandas as pd
import numpy as np
pd.options.mode.chained_assignment=None

orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
priors = pd.read_csv('../data/profile/product_brrc_profile.csv')
priors = priors[['product_id','prd_post']]
history = history.merge(orders, on='order_id', how='inner')
history = history[history['counter'] > 1]

def get_order(counter):
    order = history[history['counter'] == counter]
    order = order.groupby('user_id')['product_id'].apply(list).reset_index()
    order.columns = ['user_id'] + ['order_' + str(counter)]
    print(order.shape)
    return order

order2 = get_order(2)
order3 = get_order(3)
order4 = get_order(4)

def f1_score(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = cross_size / len(y_pred)
    r = cross_size / len(y_true)
    return 2 * p * r / (p + r)

def f1_full(y_true, y_pred):
    return [f1_score(x, y) for x, y in zip(y_true, y_pred)]

def get_fscore(data1, data2, order1, order2, feature):
    data = data1.merge(data2, on='user_id', how='left')
    data[feature] = f1_full(data[order1], data[order2])
    return data[['user_id',feature]]

f1_23 = get_fscore(order2, order3, 'order_2', 'order_3', 'fscore_23')
f1_34 = get_fscore(order3, order4, 'order_3', 'order_4', 'fscore_34')
f1_24 = get_fscore(order2, order4, 'order_2', 'order_4', 'fscore_24')

fscore = f1_23.merge(f1_34, on='user_id')
fscore = fscore.merge(f1_24, on='user_id')
fscore['fmean'] = np.mean([fscore['fscore_23'], fscore['fscore_34'], fscore['fscore_24']], axis=0)
fscore.to_csv('../data/profile/user_fscore_profile.csv', index=False)
