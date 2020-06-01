import pandas as pd
import numpy as np
import csv
from sklearn.metrics import roc_curve, auc
from scipy.sparse import coo_matrix
from implicit.nearest_neighbours import BM25Recommender, bm25_weight
from implicit.nearest_neighbours import CosineRecommender
from implicit.nearest_neighbours import TFIDFRecommender

order_products = pd.read_csv('../data/driver/driver_order_products.csv')
print(order_products.shape)
orders = pd.read_csv('../data/driver/driver_order.csv')[['user_id','order_id','counter']]
orders = orders[orders['counter'] > 2]
order_products = orders.merge(order_products, on=['order_id'], how='inner')
print(order_products.shape)
data = order_products.groupby(['user_id','product_id'])['reordered'].count().reset_index()
data['user_id'] = data['user_id'].astype(str)
data['product_id'] = data['product_id'].astype(str)

data['user_id'] = data['user_id'].astype("category")
data['product_id'] = data['product_id'].astype("category")
data['reordered'] = data['reordered'].astype(float)
matrix = coo_matrix((data['reordered'],(data['product_id'].cat.codes.copy(), data['user_id'].cat.codes.copy())))

products = pd.read_csv('../data/driver/driver_product.csv')
lookup = products.set_index('product_id')['department_id'].to_dict()
dir_mapping = dict(enumerate(data['product_id'].cat.categories))
inv_mapping = {v:k for k,v in dir_mapping.items()}

def accuracy(key, values):
    key = lookup[int(key)]
    correct = 0.
    for value in values:
        if lookup[int(dir_mapping[value[0]])] == key:
            correct += 1
    return correct / len(values)

sample = np.random.choice(products['product_id'].astype(str).values,10)
model1 = TFIDFRecommender()
model2 = CosineRecommender()
model3 = BM25Recommender(K=100, K1=1.2, B=0.75)
model1.fit(matrix)
model2.fit(matrix)
model3.fit(matrix)

for model in [model1, model2, model3]:
    scores = [accuracy(x ,model.similar_items(inv_mapping[x], N=10)) for x in sample]
    print('score: ', np.round([np.mean(scores), np.std(scores)],2))

user_mapping = dict(enumerate(data['user_id'].cat.categories))
product_mapping = dict(enumerate(data['product_id'].cat.categories))
inv_mapping = {v:k for k,v in user_mapping.items()}
print(len(user_mapping), len(product_mapping))

def recommend(userid, user_items, score_items, similarity):
    liked_vector = user_items[inv_mapping[userid]]
    recommendations = liked_vector.dot(similarity)
    recommendations = sorted(zip(recommendations.indices, recommendations.data))
    scores = []
    for item in recommendations:
        if product_mapping[item[0]] in score_items:
            scores += [[userid, product_mapping[item[0]], item[1]]]
    return scores

evaluate = matrix.T.tocsr()
driver = pd.read_csv('../data/model/dependent/dependent_n.csv')[['user_id','product_id']]
driver = driver.astype(str)
driver = driver.groupby('user_id')['product_id'].apply(list).reset_index()

def score(model):
    scores = []
    i = 0
    for user, product in zip(driver['user_id'], driver['product_id']):
        i += 1
        scores += recommend(str(user),evaluate, product, model.similarity)
        if max(i,1) % 100000 == 0:
            print(i, 'users scored...')
    return scores

tfidf_scores = pd.DataFrame(score(model1), columns = ['user_id', 'product_id', 'tfifd_score'])
cosine_scores = pd.DataFrame(score(model2), columns = ['user_id', 'product_id', 'cosine_score'])
bm25_scores = pd.DataFrame(score(model3), columns = ['user_id', 'product_id', 'bm25_score'])

similarity_score = tfidf_scores.merge(cosine_scores, on=['user_id', 'product_id'], how='outer')
similarity_score = similarity_score.merge(bm25_scores, on=['user_id', 'product_id'], how='outer')
similarity_score = similarity_score.fillna(0.)
print(similarity_score.shape)
similarity_score.to_csv('../data/similarity/similarity_score.csv', index=False)

target = pd.read_csv('../data/model/dependent/dependent_n.csv')
similarity_score = pd.read_csv('../data/similarity/similarity_score.csv')
target = target.merge(similarity_score, on=['user_id','product_id'], how='inner')
target = target[target['eval_set'] != ' test']

for feat in similarity_score.columns[2:]:
    fpr, tpr, thresholds = roc_curve(target['reordered'], target[feat])
    print('feat:', feat, 'auc:', round(100*(2*auc(fpr,tpr) - 1),2))