import numpy as np
import pandas as pd
from multiprocessing import Pool

def f1_full(y_true, y_pred):
    return np.mean([f1_score(x, y) for x, y in zip(y_true, y_pred)])

def f1_score(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = cross_size / len(y_pred)
    r = cross_size / len(y_true)
    return 2 * p * r / (p + r)

class F1Optimizer():
    
    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(args):
        user = args[0]
        P = args[1]
        pNone = args[2]
        expectations = F1Optimizer.get_expectations(P, pNone)
        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]
        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]
        return user, best_k, predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)

def append(products, none):
    if none == True:
        return [0] + products
    else:
        return products

optim = F1Optimizer().maximize_expectation

data = pd.read_csv('../data/model/score/score_n.csv')
data = data.sort_values(by=['user_id','score'], ascending = [True,False])
data = data[data['eval_set'] == 'valid']
not_none = data[data['product_id'] != 0]
none = data[data['product_id'] == 0]

list_not_none = not_none.groupby('user_id')['score'].apply(list).reset_index()
list_none = none[['user_id','score']].rename(columns={'score':'none'})
list_not_none = list_not_none.merge(list_none, on='user_id', how='inner')
list_not_none = list_not_none.to_records(index=False)
list_not_none = list(list_not_none)

pool = Pool(31)
results = pool.map(optim, list_not_none)

thresholds = pd.DataFrame(results, columns=['user_id','cutoff','none','f1'])
thresholds = not_none.merge(thresholds, on='user_id', how='left')
thresholds['rank'] = thresholds.groupby('user_id')['score'].rank(ascending=False)
thresholds = thresholds[thresholds['rank'] <= thresholds['cutoff']]
thresholds = thresholds.groupby(['user_id','none'])['product_id'].apply(list)
thresholds = thresholds.reset_index()

thresholds['product_id'] = thresholds.apply(lambda x : append(x['product_id'], x['none']), axis=1)
thresholds = thresholds[['user_id','product_id']].rename(columns={'product_id':'predicted'})
actuals = data[data['reordered'] == 1]
actuals = actuals.groupby('user_id')['product_id'].apply(list).reset_index()
actuals = actuals.rename(columns={'product_id':'actuals'})
actuals = actuals.merge(thresholds, on='user_id', how='left')
actuals['predicted'] = actuals['predicted'].map(lambda x : [0] if np.isnan(x).any() else x)

print(f1_full(actuals['actuals'],actuals['predicted']))

data = pd.read_csv('../data/model/score/score_n.csv')
data = data.sort_values(by=['user_id','score'], ascending = [True,False])
data = data[data['eval_set'] == 'test']
not_none = data[data['product_id'] != 0]
none = data[data['product_id'] == 0]
list_not_none = not_none.groupby('user_id')['score'].apply(list).reset_index()
list_none = none[['user_id','score']].rename(columns={'score':'none'})
list_not_none = list_not_none.merge(list_none, on='user_id', how='inner')
list_not_none = list_not_none.to_records(index=False)
list_not_none = list(list_not_none)
print(len(list_not_none))

pool = Pool(63)
results = pool.map(optim, list_not_none)

thresholds = pd.DataFrame(results, columns=['user_id','cutoff','none','f1'])
thresholds = not_none.merge(thresholds, on='user_id', how='left')
thresholds['rank'] = thresholds.groupby('user_id')['score'].rank(ascending=False)
thresholds = thresholds[thresholds['rank'] <= thresholds['cutoff']]
thresholds = thresholds.groupby(['user_id','none'])['product_id'].apply(list)
thresholds = thresholds.reset_index()
thresholds['product_id'] = thresholds.apply(lambda x : append(x['product_id'], x['none']), axis=1)
thresholds = thresholds[['user_id','product_id']].rename(columns={'product_id':'predicted'})
actuals = data[['user_id']].drop_duplicates()
actuals = actuals.merge(thresholds, on='user_id', how='left')
actuals['predicted'] = actuals['predicted'].map(lambda x : [0] if np.isnan(x).any() else x)
actuals['predicted'] = actuals['predicted'].map(lambda x : [str(y) for y in x])
actuals['predicted'] = actuals['predicted'].map(lambda y : ['None' if x == '0' else x for x in y])
actuals['products'] = actuals['predicted'].map(lambda x : ' '.join([str(y) for y in x]))
orders = pd.read_csv('../data/driver/driver_order.csv')
orders = orders[orders['eval_set'] == 'test'][['order_id','user_id']]
score = orders.merge(actuals, on='user_id', how='inner').drop('user_id',axis=1)
score[['order_id','products']].to_csv('../data/model/submission/submit_v4.csv', index=False)
print(score.shape)