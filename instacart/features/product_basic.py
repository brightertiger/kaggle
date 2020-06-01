import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
vectorizer = CountVectorizer()
tfidf = TfidfTransformer(norm="l2", smooth_idf=True)

product_profile = pd.read_csv('../data/driver/driver_product.csv')
names = pd.read_csv('../data/download/products.csv')[['product_id','product_name']]
orders = pd.read_csv('../data/driver/driver_order.csv')
orders = orders[orders['counter'] > 1]
products = pd.read_csv('../data/driver/driver_order_products.csv')
users = pd.read_csv('../data/driver/driver_user.csv')
products = products.merge(orders, on='order_id', how='inner').drop('eval_set',axis=1)
products = products.merge(users, on='user_id', how='inner')
print(products.shape)

def common_profile(data, level, prefix):
    aggregate = {}
    aggregate['reordered'] = [np.sum, np.mean]
    aggregate['order_id'] = pd.Series.nunique
    aggregate['user_id'] = pd.Series.nunique
    aggregate['order_number'] = np.median
    aggregate['add_to_cart_order'] = np.median
    data = data.groupby(level).agg(aggregate).reset_index()
    features = ['sum_rdr', 'avg_rdr','cds_ord','cds_usr','med_ordn' ,'med_addcrt']
    data.columns = [level] + [prefix + x for x in features]
    data[prefix + 'rt_ord_usr'] = data[prefix + 'cds_ord'] / data[prefix + 'cds_usr']
    return data

common_product = common_profile(products,'product_id','prd_')
common_department = common_profile(products,'department_id','dep_')
common_aisle = common_profile(products,'aisle_id','ais_')

prod_last_order = products.groupby(['user_id','product_id'])['order_number'].max().reset_index()
user_last_order = products.groupby(['user_id'])['order_number'].max().reset_index()
prod_last_order.columns = ['user_id','product_id','prod_last_order']
user_last_order.columns = ['user_id','user_last_order']
affinity = prod_last_order.merge(user_last_order, on='user_id')
affinity['order_since'] = affinity['user_last_order'] - affinity['prod_last_order']
prod_affinity = affinity.groupby('product_id')['order_since'].apply(np.median).reset_index()
prod_affinity.columns = ['product_id','product_affinity']

def clean_string(name):
    name = name.replace('-',' ')
    name = name.replace('&',' ')
    name = name.replace("'N",' ')
    string = name.split()
    string = [''.join(filter(str.isalpha,x.lower())) for x in string]
    return ' '.join(string)

prod_name = names.copy()
prod_name['length'] = prod_name['product_name'].map(lambda x : len(x))
prod_name['words'] = prod_name['product_name'].map(lambda x : len(x.split()))
prod_name['organic'] = prod_name['product_name'].map(lambda x : 1 if 'organic' in x.lower() else 0)
clean_name = prod_name['product_name'].map(clean_string)
clean_name = list(prod_name['product_name'].map(clean_string))
vec_name = vectorizer.fit_transform(clean_name)
vec_idf = tfidf.fit_transform(vec_name)
vec_idf = vec_idf.todense()
prod_name['mean_tfidf'] = vec_idf.sum(axis=1)
prod_name['mean_tfidf'] = prod_name['mean_tfidf'] / prod_name['length']
prod_name['max_tfidf'] = vec_idf.max(axis=1)
prod_name = prod_name.drop('product_name',axis=1)

product_profile = product_profile.merge(common_product, on='product_id', how='left')
product_profile = product_profile.merge(common_department, on='department_id', how='left')
product_profile = product_profile.merge(common_aisle, on='aisle_id', how='left')
product_profile = product_profile.merge(prod_affinity, on='product_id', how='left')
product_profile = product_profile.merge(prod_name, on='product_id', how='left')
product_profile = product_profile.fillna(0.)
product_profile = product_profile.drop(['department_id', 'aisle_id'], axis=1)
product_profile.to_csv('../data/profile/product_basic_profile.csv', index=False)

target = pd.read_csv('../data/model/dependent/dependent_n.csv')
target = target.merge(product_profile, on='product_id', how='inner')

for feat in product_profile.columns[1:]:
    fpr, tpr, thresholds = roc_curve(target['reordered'], target[feat])
    print('feat:', feat, 'auc:', round(100*(2*auc(fpr,tpr) - 1),2))