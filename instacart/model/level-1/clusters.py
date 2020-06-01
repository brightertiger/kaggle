import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import fisher_exact

data = pd.read_csv('../data/gensim/prodvecs.csv')
model = KMeans(n_clusters = 30, precompute_distances = True, n_init=10, n_jobs = 4)
model.fit(data.iloc[:,1:])
data['prd_label'] = model.predict(data.iloc[:,1:])
data[['product_id','prd_label']].to_csv('../data/gensim/cluster_product.csv', index=False)
labels = data[['product_id','prd_label']]
dep = pd.read_csv('../data/model/dependent/dependent_n.csv')
dep = dep.merge(labels, on='product_id', how='inner')
cluster = pd.get_dummies(data=dep['prd_label'], prefix='prd_cls')
dep = dep.join(cluster)
dep =dep[dep['eval_set'] == 'valid']

for feat in reversed(['prd_cls_' + str(x) for x in range(30)]):
    oddsratio = pd.crosstab(dep[feat], dep['reordered']).reset_index(drop=True)
    oddsratio = np.mat(oddsratio)
    cross  = oddsratio[1,1]
    oddsratio = (oddsratio[0,0]*oddsratio[1,1]) / (oddsratio[1,0]*oddsratio[0,1])
    if cross > 2000 and (oddsratio < 0.80 or oddsratio > 1.2):
        print(feat, dep[feat].sum(), cross, round(oddsratio,4))

data = pd.read_csv('../data/gensim/uservecs.csv')
model = KMeans(n_clusters = 50, precompute_distances = True, n_init=10, n_jobs = 4)
model.fit(data.iloc[:,1:])
data['usr_label'] = model.predict(data.iloc[:,1:])
data[['user_id','usr_label']].to_csv('../data/gensim/cluster_user.csv', index=False)

labels = data[['user_id','usr_label']]
dep = pd.read_csv('../data/model/dependent/dependent_n.csv')
dep = dep.merge(labels, on='user_id', how='inner')
cluster = pd.get_dummies(data=dep['usr_label'], prefix='prd_cls')
dep = dep.join(cluster)
dep =dep[dep['eval_set'] == 'valid']

for feat in reversed(['prd_cls_' + str(x) for x in range(50)]):
    oddsratio = pd.crosstab(dep[feat], dep['reordered']).reset_index(drop=True)
    oddsratio = np.mat(oddsratio)
    cross  = oddsratio[1,1]
    oddsratio = (oddsratio[0,0]*oddsratio[1,1]) / (oddsratio[1,0]*oddsratio[0,1])
    if cross > 1000 and (oddsratio <= 0.80 or oddsratio >= 1.2):
        print(feat, dep[feat].sum(), cross, round(oddsratio,4))