import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression

# random = 0.26
# contant = 0.26

def evaluate(feature):
    data = pd.read_csv('../data/download/train.csv')[['item_id','deal_probability']]
    data = data.merge(feature, on='item_id').fillna(-1.)
    low = data.iloc[:,-1].quantile(0.01)
    high = data.iloc[:,-1].quantile(0.99)
    data.iloc[:,-1] = data.iloc[:,-1].clip(low, high)
    model = f_regression(np.array(data.iloc[:,-1])[:,np.newaxis], data['deal_probability'])
    print('score:', model)
    return None

feature = pd.read_csv('../data/data/features/image/vgg_train.csv')
feature = feature.rename(columns={'deal_probability':'vgg_model'})
evaluate(feature)