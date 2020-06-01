import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import warnings
warnings.simplefilter('ignore')

orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
history = history[history['counter'] > 1]
del orders

def subsequence(nums):
    return max(np.split(nums, np.where(np.diff(nums) != 1)[0]+1), key=len).tolist()

product_hist = history[['user_id','product_id','counter']]
product_hist = product_hist.sort_values(by=['user_id','product_id','counter'])
product_hist = product_hist.groupby(['user_id','product_id'])['counter'].apply(list)
product_hist = product_hist.reset_index()

product_hist['sublist'] = product_hist['counter'].map(lambda x : subsequence(x))
product_hist['sequence'] = product_hist['sublist'].map(lambda x : len(x))
product_hist['total'] = product_hist['counter'].map(lambda x : len(x))
product_hist['minimum'] = product_hist['counter'].map(lambda x : min(x))
product_hist['seq_differ'] = product_hist['counter'].map(lambda x : np.mean(np.diff(x))) 

prod_diff = product_hist.groupby('product_id')['seq_differ'].mean().reset_index()
prod_diff = prod_diff.rename(columns={'differ':'prd_differ'})
product_hist = product_hist.merge(prod_diff, on='product_id', how='left')

product_hist['score_seq1'] = product_hist['sequence'] / product_hist['total']
product_hist['score_seq2'] = product_hist['sequence'] / product_hist['minimum']
product_hist['score_seq3'] = (product_hist['minimum'] - 1) / product_hist['seq_differ']
product_hist['score_seq4'] = (product_hist['minimum'] - 1) / product_hist['prd_differ']
product_hist = product_hist.drop(['counter','sublist','total','minimum'], axis=1)
user_prd_seq = history[['user_id','product_id']].drop_duplicates()
user_prd_seq = user_prd_seq.merge(product_hist, on=['user_id','product_id'], how='left')

user_prd_seq.to_csv('../data/profile/user_product_max_seq.csv', index=False)

orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
history = history[history['counter'] > 1]
del orders

def gap_analysis(array):
    if len(array) == 0:
        return (0,0)
    else:
        max_gap = 0
        avg_gap = 0
        for i in range(len(array)-1):
            _max = max(array[i])
            _min = min(array[i+1])
            gap = _min - _max
            if gap >= max_gap:
                max_gap = gap
            avg_gap += gap
        return (max_gap, avg_gap / len(array))

def current_sequence(nums):
    total = len(nums)
    nums = list(np.split(nums, np.where(np.diff(nums) != 1)[0] + 1))
    curr_streak = len(nums[0])
    num_streaks = len(nums)
    avg_length = np.mean([len(x) for x in nums])
    max_gap, avg_gap = gap_analysis(nums)
    ratio_length = num_streaks / total
    return [curr_streak, num_streaks, avg_length, ratio_length, avg_gap, max_gap]

product_hist = history[['user_id','product_id','counter']]
product_hist = product_hist.sort_values(by=['user_id','product_id','counter'])
product_hist = product_hist.groupby(['user_id','product_id'])['counter'].apply(list)
product_hist = product_hist.reset_index()

product_hist['analysis'] = product_hist['counter'].map(lambda x : current_sequence(x))
product_hist['curr_streak'] = product_hist['analysis'].map(lambda x : x[0])
product_hist['num_streak'] = product_hist['analysis'].map(lambda x : x[1])
product_hist['avg_length'] = product_hist['analysis'].map(lambda x : x[2])
product_hist['ratio_length'] = product_hist['analysis'].map(lambda x : x[3])
product_hist['avg_gap'] = product_hist['analysis'].map(lambda x : x[4])
product_hist['max_gap'] = product_hist['analysis'].map(lambda x : x[5])
product_hist = product_hist.drop(['counter','analysis'], axis = 1)
product_hist.to_csv('../data/profile/user_product_curr_seq.csv', index=False)
orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
history = history[history['counter'] > 1]
del orders

counters = history[['user_id','product_id','counter']]
counters = counters.sort_values(by=['user_id','product_id','counter'])
counters['recent'] = 1 / np.log2(counters['counter'])

def log_bound(x):
    value = 0.
    for y in range(2,x):
        value += 1 / np.log2(y)
    return value

bound = counters.groupby('user_id')['counter'].max().reset_index()
bound['upper_bound'] = bound['counter'].map(log_bound)
bound = bound[['user_id','upper_bound']]

counters = counters.merge(bound, on='user_id')
counters = counters.groupby(['user_id','product_id','upper_bound'])['recent'].sum().reset_index()
counters = counters.rename(columns={'recent' : 'recency_counter'})
counters['recency_counter_bound'] = counters['recency_counter'] / counters['upper_bound']
counters = counters.drop('upper_bound', axis=1)

orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
history = history[history['counter'] > 1]

orders = history[['user_id','product_id','order_number']]
orders = orders.sort_values(by=['user_id','product_id','order_number'])
orders['recency_order1'] = orders['order_number']
orders['recency_order2'] = orders['order_number']**2
bound = orders.groupby('user_id')['order_number'].max().reset_index()
orders = orders.groupby(['user_id','product_id'])[['recency_order1','recency_order2']].sum().reset_index()
bound['upper_bound1'] =  bound['order_number'].map(lambda x : 0.5*x*(x+1))
bound['upper_bound2'] =  bound['order_number'].map(lambda x : (x*(x+1)*(2*x+1))/6)
recent = orders.merge(bound, on=['user_id'])
recent['recency_order1_bound'] = recent['recency_order1'] / recent['upper_bound1']
recent['recency_order2_bound'] = recent['recency_order2'] / recent['upper_bound2']
recent['recency_order_ratio'] = recent['recency_order2_bound'] / recent['recency_order1_bound']
recent = recent.drop(['upper_bound1','upper_bound2'],axis=1)

orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
history = history[history['counter'] > 1]
del orders

skew = history.groupby(['user_id','product_id'])['order_number'].agg(['mean','max'])
skew = skew.reset_index()
skew['skew'] = skew['max'] / skew['mean']
skew = skew.drop(['mean','max'], axis=1)

orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
history = history[history['counter'] > 1]
history = history[['user_id','product_id','counter']]
history = history.sort_values(by=['user_id','product_id','counter'])
del orders

past_pattern = history.copy()
past_pattern['last_counter'] = past_pattern.groupby(['user_id','product_id'])['counter'].shift(1)
past_pattern = past_pattern[np.logical_not(past_pattern['last_counter'].isnull())]
past_pattern['diff'] = past_pattern['counter'] - past_pattern['last_counter']
past_pattern = past_pattern.groupby(['user_id','product_id'])['diff'].mean().reset_index()
past_pattern = past_pattern.rename(columns={'diff':'pattern'})

last_order = history.groupby(['user_id','product_id'])['counter'].min().reset_index()
last_order['order_since'] = last_order['counter'] - 1
last_order = last_order[['user_id','product_id','order_since']]

pattern = last_order.merge(past_pattern, on=['user_id','product_id'], how='left')
pattern['likelihood_order'] = pattern['order_since'] / pattern['pattern']  
pattern_orders = pattern[['user_id','product_id','likelihood_order']]

orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
history = history.merge(orders, on='order_id', how='inner')
history = history[history['counter'] > 1]
orders = orders[['user_id','days_since_prior_order','counter']]
days = orders.sort_values(by=['user_id','counter'])
days['days'] = days.groupby('user_id')['days_since_prior_order'].shift(1)
days = days[days['counter'] >  1]
days['cum_days'] = days.groupby('user_id')['days'].cumsum()
days = days[['user_id','counter','cum_days']]
history = history.merge(days, on=['user_id','counter'])
history = history[['user_id','product_id','counter','cum_days']]
history = history.sort_values(by=['user_id','product_id','counter'])
history['weight'] = np.exp(-1.*history['cum_days']/30.)
decay = history.groupby(['user_id','product_id'])['weight'].sum().reset_index()
decay = decay.rename(columns={'weight':'decay'})

orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
orders = orders[['user_id', 'order_id','days_since_prior_order', 'counter']]
orders = orders.sort_values(by=['user_id','counter'])
orders['days_since_prior_order'] = orders.groupby(['user_id'])['days_since_prior_order'].shift(1)
orders = orders[orders['counter'] > 1]
orders['cum_days'] = orders.groupby(['user_id'])['days_since_prior_order'].cumsum()
orders = orders[['user_id','order_id','counter','cum_days']]
history = history.merge(orders, on='order_id', how='inner')
history = history.sort_values(by=['user_id','product_id', 'counter','cum_days'])
history = history[['user_id','product_id', 'counter','cum_days']]
del orders

past_pattern = history.copy()
past_pattern['last_counter'] = past_pattern.groupby(['user_id','product_id'])['cum_days'].shift(1)
past_pattern = past_pattern[np.logical_not(past_pattern['last_counter'].isnull())]
past_pattern['diff'] = past_pattern['cum_days'] - past_pattern['last_counter']
past_pattern = past_pattern.groupby(['user_id','product_id'])['diff'].mean().reset_index()
past_pattern = past_pattern.rename(columns={'diff':'pattern'})

last_order = history.groupby(['user_id','product_id'])['cum_days'].min().reset_index()
last_order['order_since'] = last_order['cum_days']
last_order = last_order[['user_id','product_id','order_since']]

pattern = last_order.merge(past_pattern, on=['user_id','product_id'], how='left')
pattern['likelihood_days'] = pattern['order_since'] / pattern['pattern'].clip_lower(1) 
pattern_days = pattern[['user_id','product_id','likelihood_days']]

print(counters.shape, recent.shape, decay.shape)
print(skew.shape, pattern_orders.shape, pattern_days.shape)

recency = counters.merge(recent, on=['user_id','product_id'])
recency = recency.merge(decay, on=['user_id','product_id'])
recency = recency.merge(skew, on=['user_id','product_id'])
recency = recency.merge(pattern_orders, on=['user_id','product_id'])
recency = recency.merge(pattern_days, on=['user_id','product_id'])
recency = recency.drop('order_number', axis=1)
recency.to_csv('../data/profile/user_product_time.csv', index=False)