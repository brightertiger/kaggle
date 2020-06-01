import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
pd.options.mode.chained_assignment = None

transition = pd.read_csv('../data/similarity/transition_matrix.csv')
orders = pd.read_csv('../data/driver/driver_order.csv')
dependent = pd.read_csv('../data/model/dependent/dependent_n.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
prior = pd.read_csv('../data/profile/product_brrc_profile.csv')[['product_id','prd_post']]
history = history.merge(orders, on='order_id')
history = history[history['counter'] > 1]

state = history[history['counter'] <= 6][['order_id','user_id','product_id','counter']]
state['counter'] = state['counter'] - 1
state = dependent.merge(state, on=['user_id','product_id'], how='outer',indicator=True)
state = state[['order_id','user_id','product_id','counter']]
state = state.pivot_table(index=['user_id','product_id'], columns='counter', values='order_id', aggfunc='count')
state = state.reset_index()
state.columns = ['user_id','product_id','1','2','3','4','5']
state = state.fillna(0.)

none = dependent[['user_id','product_id']].drop_duplicates()
none = none.merge(state[['user_id','product_id']], on =['user_id','product_id'], how = 'left', indicator=True)
none = none[none['_merge'] == 'left_only'][['user_id','product_id']]
none['1'] = 0
none['2'] = 0
none['3'] = 0
none['4'] = 0
none['5'] = 0

state = state.append(none)
transition_1 = transition[['product_id','1','mc1']].drop_duplicates()
transition_2 = transition[['product_id','1','2','mc2']].drop_duplicates()
transition_3 = transition[['product_id','1','2','3','mc3']].drop_duplicates()
transition_4 = transition[['product_id','1','2','3','4','mc4']].drop_duplicates()
transition_5 = transition[['product_id','1','2','3','4','5','mc5']].drop_duplicates()

user_transition = state.merge(transition_1, on=['product_id','1'], how='left')
user_transition = user_transition.merge(transition_2, on=['product_id','1','2'], how='left')
user_transition = user_transition.merge(transition_3, on=['product_id','1','2','3'], how='left')
user_transition = user_transition.merge(transition_4, on=['product_id','1','2','3','4'], how='left')
user_transition = user_transition.merge(transition_5, on=['product_id','1','2','3','4','5'], how='left')

user_transition['usr_prd_l1'] = user_transition['1']
user_transition['usr_prd_l2'] = user_transition['usr_prd_l1'] + user_transition['2']
user_transition['usr_prd_l3'] = user_transition['usr_prd_l2'] + user_transition['3']
user_transition['usr_prd_l4'] = user_transition['usr_prd_l3'] + user_transition['4']
user_transition['usr_prd_l5'] = user_transition['usr_prd_l4'] + user_transition['5']

user_transition = user_transition.merge(prior, on='product_id', how='inner')

user_transition['mcrt1'] = user_transition['mc1'] / user_transition['prd_post'].clip_lower(1e-5)
user_transition['mcrt2'] = user_transition['mc2'] / user_transition['prd_post'].clip_lower(1e-5)
user_transition['mcrt3'] = user_transition['mc3'] / user_transition['prd_post'].clip_lower(1e-5)
user_transition['mcrt4'] = user_transition['mc4'] / user_transition['prd_post'].clip_lower(1e-5)
user_transition['mcrt5'] = user_transition['mc5'] / user_transition['prd_post'].clip_lower(1e-5)

user_transition = user_transition.drop(['1','2','3','4','5','prd_post'], axis=1)
user_transition.to_csv('../data/profile/user_product_transition.csv',index=False)
