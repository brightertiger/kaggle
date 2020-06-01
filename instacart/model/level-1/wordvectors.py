import pandas as pd
import numpy as np
from gensim.models import doc2vec, word2vec, keyedvectors
pd.set_option('chained_assignment',None)

order_products = pd.read_csv('../data/driver/driver_order_products.csv').drop('reordered',axis=1)
orders = pd.read_csv('../data/driver/driver_order.csv')[['order_id','user_id']]
data = orders.merge(order_products, on='order_id')
data = data.sort_values(by=['order_id','add_to_cart_order'])
print(order_products.shape, orders.shape, data.shape)

prefix = np.array(['p_'] * data.shape[0])
data['product_id'] = np.core.defchararray.add(prefix, data['product_id'].values.astype(np.str))
prefix = np.array(['d_'] * data.shape[0])
data['department_id'] = np.core.defchararray.add(prefix, data['department_id'].values.astype(np.str))
prefix = np.array(['a_'] * data.shape[0])
data['aisle_id'] = np.core.defchararray.add(prefix, data['aisle_id'].values.astype(np.str))
sep = [','] * data.shape[0]
data['product'] = data['product_id'] + sep + data['aisle_id'] + sep + data['department_id']
data = data[['order_id','user_id','product']]
data = data.groupby(['user_id','order_id'])['product'].apply(','.join).reset_index()

corpus = []
for line in list(data['product'].values):
    corpus += [line.split(',')]

products = pd.read_csv('../data/driver/driver_product.csv')
lookup = products.set_index('product_id')['department_id'].to_dict()

def accuracy(key, values):
    key = lookup[int(key[2:])]
    correct = 0.
    for value in values:
        try:
            if lookup[int(value[0][2:])] == key:
                correct += 1
        except KeyError:
            pass
    return correct / len(values)    

params = {}
params['size'] = 16
params['window'] = 8 
params['min_count'] = 0
params['sample'] = 1e-4 
params['negative'] = 20
params['workers'] = 4
params['hs'] = 0
params['seed'] = 108

model = word2vec.Word2Vec(**params, iter=1)
model.build_vocab(corpus)

words = list(model.wv.vocab.keys())
sample = np.random.choice(words,5000)
score = np.mean([accuracy(x ,model.most_similar([x])) for x in sample])
print('start score:', score)

alpha = 0.1

for epoch in range(7):
    model.train(corpus, total_examples=model.corpus_count, epochs=1, start_alpha=alpha, end_alpha=alpha)
    scores = [accuracy(x ,model.most_similar([x])) for x in sample]
    print('alpha:', round(alpha,4),'score:', np.round([np.mean(scores), np.std(scores)],2))
    alpha = alpha * 0.95

model.wv.save_word2vec_format('../data/gensim/wordvectors.txt', binary=False)

wordvecs = pd.read_csv('../data/gensim/wordvectors.txt', sep=' ', header = None, skiprows=1)
wordvecs.columns = ['id'] + ['pv_' + str(x) for x in range(16)]
prodvecs = wordvecs[wordvecs['id'].str[:2] == 'p_']
prodvecs['product_id'] = prodvecs['id'].map(lambda x : int(x.split('_')[1]))
prodvecs = prodvecs.groupby('product_id').mean().reset_index()
prodvecs.columns = ['product_id'] + ['prdwv_' + str(x) for x in range(16)]
print(prodvecs.shape)
prodvecs.to_csv('../data/gensim/prodvecs.csv', index=False)