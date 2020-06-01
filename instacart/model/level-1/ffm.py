import pandas as pd
import numpy as np
from joblib import Parallel, delayed
pd.options.mode.chained_assignment = None

orders = pd.read_csv('../data/driver/driver_order.csv').drop(['eval_set'], axis=1)
history = pd.read_csv('../data/driver/driver_order_products.csv')
products = pd.read_csv('../data/driver/driver_product.csv')
features = ['eval_set','reordered','user_id','product_id','aisle_id','department_id']
features += ['order_dow','order_hour_of_day','order_number','days_since_prior_order']
dependent = pd.read_csv('../data/model/dependent/dependent_n_1.csv')
orders = orders[orders['counter'] == 2] 
dependent = dependent.merge(orders, on=['user_id'], how='inner')
dependent = dependent.merge(products, on=['product_id'], how='inner')
dependent = dependent[features]
dependent['days_since_prior_order'] = dependent['days_since_prior_order'].astype(int)

orders = pd.read_csv('../data/driver/driver_order.csv').drop(['eval_set'], axis=1)
history = pd.read_csv('../data/driver/driver_order_products.csv')
products = pd.read_csv('../data/driver/driver_product.csv')
history = orders.merge(history, on='order_id')
independent = history[history['counter'] > 2]

user_product = independent.groupby(['user_id','product_id'])['order_id'].count().reset_index()
user_product = user_product.rename(columns={'order_id' : 'prd_cnt'})
user_product['prd_cnt'] = user_product['prd_cnt'].clip_upper(10)

user_aisle = independent.groupby(['user_id','aisle_id'])['order_id'].count().reset_index()
user_aisle = user_aisle.rename(columns={'order_id' : 'ais_cnt'})
user_aisle['ais_cnt'] = user_aisle['ais_cnt'].clip_upper(20)

user_dept = independent.groupby(['user_id','department_id'])['order_id'].count().reset_index()
user_dept = user_dept.rename(columns={'order_id' : 'dep_cnt'})
user_dept['dep_cnt'] = user_dept['dep_cnt'].clip_upper(30)

order_since = independent.groupby(['user_id','product_id'])['counter'].min().reset_index()
order_since['counter'] = order_since['counter'] - 2
order_since['counter'] = order_since['counter'].clip_upper(15)
order_since = order_since.rename(columns={'counter':'ordn_snc'})

indep_orders = orders[orders['counter'] >= 2]
indep_orders['days'] = indep_orders.groupby(['user_id'])['days_since_prior_order'].shift(1)
indep_orders = indep_orders[indep_orders['counter'] > 2]
indep_orders['cum_days'] = indep_orders.groupby(['user_id'])['days'].cumsum()
indep_orders = independent.merge(indep_orders[['order_id','cum_days']], on='order_id')
days_since = indep_orders.groupby(['user_id','product_id'])['cum_days'].min().reset_index()
days_since['cum_days'] = days_since['cum_days'].clip_upper(120)
days_since = days_since.rename(columns={'cum_days':'prd_dysc'})
del indep_orders

last5 = independent[independent['counter'] <= 7]
last5['counter'] = last5['counter'] - 2
last5 = last5.groupby(['user_id','product_id'])['order_id'].count().reset_index()
last5 = last5.rename(columns={'order_id':'last5'})

del independent
dependent = dependent.merge(user_product, on=['user_id','product_id'], how='left')
del user_product
dependent = dependent.merge(user_aisle, on=['user_id','aisle_id'], how='left')
del user_aisle
dependent = dependent.merge(user_dept, on=['user_id','department_id'], how='left')
del user_dept
dependent = dependent.merge(order_since, on=['user_id','product_id'], how='left')
del order_since
dependent = dependent.merge(days_since, on=['user_id','product_id'], how='left')
del days_since
dependent = dependent.merge(last5, on=['user_id','product_id'], how='left')
del last5
dependent['reordered'] = dependent['reordered'].astype(int)
dependent = dependent.fillna(0)

values = pd.DataFrame([])
    
def value_map(variable, field):
    global dependent
    temp = dependent[[variable]].drop_duplicates()
    temp['feature'] = variable
    temp['value'] = temp[variable]
    temp['field'] = field
    global values
    values = values.append(temp[['feature','value','field']])
    return None

value_map('order_dow',1)
value_map('order_hour_of_day',1)
value_map('order_number',2)
value_map('days_since_prior_order',2)
value_map('prd_cnt',3)
value_map('ais_cnt',3)
value_map('dep_cnt',3)
value_map('ordn_snc',4)
value_map('prd_dysc',4)
value_map('last5',4)
value_map('department_id',5)
value_map('aisle_id',5)
value_map('product_id',5)
value_map('user_id',6)

values['value'] = values['value'].astype(int)
values = values.reset_index(drop=True)
values['index'] = values.index + 1

values.to_csv('../data/ffm/value_map.csv', index=False)

fields = values[['feature','field']].drop_duplicates()
fields = fields.set_index('feature')['field'].to_dict()

index = {}

for field in fields.keys():
    temp = values[values['feature'] == field][['value','index']]
    index[field] = temp.set_index('value')['index'].to_dict()

def libsvm_format(x):
    if x.index.values[0] % 1000000 == 0:
        print(x.index.values[0], 'rows done..')
    string = ''
    string += str(int(x['reordered'].values[0])) + ' '
    global fields
    for field in fields.keys():
        string += str(fields[field]) + ':'
        global index
        _value = x[field].values[0]
        _index = index[field][_value]
        string += str(_index) + ':'
        string += '1' + ' '
    output = []
    output += [x['eval_set'].values[0]]
    output += [x['user_id'].values[0]]
    output += [x['product_id'].values[0]]
    output += [x['reordered'].values[0]]
    output += [string.strip()]
    return output

print(dependent.shape)

dependent = dependent.groupby(['eval_set', 'user_id','product_id'])
results = Parallel(n_jobs=2)(delayed(libsvm_format)(grp.copy()) for _, grp in dependent)
results = pd.DataFrame(results, columns=['eval_set','user_id','product_id','reordered','data'])
print(results.shape)
results[['data']].to_csv('../data/ffm/data/ffm_train', index=False, header=None)
results[['eval_set','user_id','product_id','reordered']].to_csv('../data/ffm/data/ffm_train_driver.csv', index=False)