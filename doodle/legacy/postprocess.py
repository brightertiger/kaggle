import pandas as pd
import numpy as np
import pickle

def top_3(scores):
    global mapping
    scores = np.argsort(scores)[-3:]
    scores = list(reversed(scores))
    scores = [mapping[x] for x in scores]
    scores = ' '.join(scores)
    return scores.strip()

mapping = pickle.load(open('../data/categories.pkl','rb'))
mapping = [x.replace('.csv','') for x in mapping]
mapping = [x.replace(' ', '_') for x in mapping]

data = pd.read_csv('../data/score/resnet_2.csv')
data['word'] = data.iloc[:,:-1].apply(lambda x : top_3(x), axis=1)
data[['key_id','word']].to_csv('../data/submit/check.csv', index=False)