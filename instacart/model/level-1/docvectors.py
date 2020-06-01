import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import doc2vec, word2vec
pd.set_option('chained_assignment',None)

def map_user(user_id, order_id):
    if user_id in users:
        if order_id % 2 == 0:
            return 'u_' + str(user_id) + '_even'
        else:
            return 'u_' + str(user_id) + '_odd'
    else:
        return 'u_' + str(user_id)

orders = pd.read_csv('../data/driver/driver_order.csv')
orders = orders[['order_id','user_id']]
users = orders.groupby('user_id')['order_id'].apply(pd.Series.nunique).reset_index()
users = users[users['order_id'] >= 20]
users = users[users['user_id'] % 100 <= 3] 
users = users['user_id'].values
orders['user'] = orders.apply(lambda x : map_user(x['user_id'], x['order_id']), axis=1)

order_products = pd.read_csv('../data/driver/driver_order_products.csv').drop('reordered',axis=1)
print(orders.shape, order_products.shape)
data = orders.merge(order_products, on='order_id')
print(data.shape)
data = data.sort_values(by=['order_id','add_to_cart_order'])
prefix = np.array(['p_'] * data.shape[0])
data['product_id'] = np.core.defchararray.add(prefix, data['product_id'].values.astype(np.str))
prefix = np.array(['d_'] * data.shape[0])
data['department_id'] = np.core.defchararray.add(prefix, data['department_id'].values.astype(np.str))
prefix = np.array(['a_'] * data.shape[0])
data['aisle_id'] = np.core.defchararray.add(prefix, data['aisle_id'].values.astype(np.str))
sep = [','] * data.shape[0]
data['product'] = data['product_id'] + sep + data['aisle_id'] + sep + data['department_id']
data = data[['order_id','user','product']]
data = data.groupby(['user','order_id'])['product'].apply(','.join).reset_index()

corpus = []
for user, line in zip(data['user'].values, data['product'].values):
    corpus += [doc2vec.TaggedDocument(words = line.split(','), tags = [user])]

params = {}
params['size'] = 16 
params['window'] = 8 
params['min_count'] = 0 
params['sample'] = 1e-3 
params['negative'] = 20
params['workers'] = 8
params['hs'] = 0
params['seed'] = 108

model = doc2vec.Doc2Vec(**params, iter=1)
model.build_vocab(corpus)
model.intersect_word2vec_format('../data/gensim/wordvectors.txt', lockf=0.0)

def accuracy():
    scores = []
    for user in users:
            u1 = 'u_' + str(user) + '_even'
            u2 = 'u_' + str(user) + '_odd'
            scores += [model.docvecs.similarity(u1,u2)]
    return np.round([np.mean(scores), np.std(scores)], 2)

print('start score:', accuracy())
alpha = 0.1

for epoch in range(5):
    model.train(corpus, total_examples=model.corpus_count, epochs=1, start_alpha=alpha, end_alpha=alpha)
    print('alpha:', round(alpha,4), 'score:', accuracy())
    alpha = alpha * 0.95

model.save_word2vec_format('../data/gensim/docvectors.txt', binary=False, doctag_vec=True, word_vec=False, prefix='')
docvecs = pd.read_csv('../data/gensim/docvectors.txt', sep=' ', header = None, skiprows=1)
docvecs.columns = ['id'] + ['uv_' + str(x) for x in range(16)]
uservecs = docvecs[docvecs['id'].str[:5] == '*dt_u']
uservecs['user_id'] = uservecs['id'].map(lambda x : int(x.split('_')[2]))
uservecs = uservecs.drop('id',axis=1)
uservecs = uservecs.groupby('user_id').mean().reset_index()
uservecs.columns = ['user_id'] + ['usrwv_' + str(x) for x in range(16)]
print(uservecs.shape)
uservecs.to_csv('../data/gensim/uservecs.csv', index=False)