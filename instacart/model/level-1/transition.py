import pandas as pd
import numpy as np
pd.options.mode.chained_assignment=None

orders = pd.read_csv('../data/driver/driver_order.csv')
history = pd.read_csv('../data/driver/driver_order_products.csv')
priors = pd.read_csv('../data/profile/product_brrc_profile.csv')
priors = priors[['product_id','prd_post']]
history = history.merge(orders, on='order_id', how='inner')
history = history[history['counter'] > 1]

def barreca(posterior,prior,n,k=75,f=5):
    factor = np.exp(min(n-k,1000)/f)
    factor = factor / (factor + 1)
    if np.isnan(factor):
        factor = 1.
    return factor * posterior + (1 - factor) * prior

def get_dependent(counter):
    candidate = history[history['counter'] >= counter + 1]
    candidate = candidate[['user_id','product_id']].drop_duplicates()
    actuals = history[history['counter'] == counter]
    actuals = actuals[['user_id','product_id']]
    data = candidate.merge(actuals, on=['user_id','product_id'], how='left', indicator=True)
    data['reordered'] = data['_merge'].map(lambda x : 1 if x == 'both' else 0)
    return data[['user_id','product_id','reordered']]

def transition_matrix(dim, counter):
    if counter % 5 == 0:
        print(counter, 'status completed...')
    independent = history[history['counter'] >= counter + 1]
    independent = independent[independent['counter'] <= counter + dim]
    independent['counter'] = independent['counter'] - [counter] * independent.shape[0]
    independent['counter'] = independent['counter'].astype(str)
    independent = independent[['order_id','user_id','product_id','counter']]
    index = ['user_id','product_id']
    independent = independent.pivot_table(index=index, columns=['counter'], values=['order_id'], aggfunc='count')
    independent.columns = independent.columns.droplevel(0)
    independent = independent.fillna(0).reset_index()
    dependent = get_dependent(counter)
    data = dependent.merge(independent, on=['user_id','product_id'], how='outer', indicator=True)
    data = data.drop('_merge',axis=1)
    data = data.fillna(0)
    index = ['product_id'] + [str(x) for x in range(1,dim+1)]
    data = data.pivot_table(index=index, columns='reordered', values='user_id', aggfunc='count')
    data = data.reset_index().fillna(0)
    data['positive'] = data[1]
    data['negative'] = data[0]
    data = data.drop([0,1],axis=1)
    return data

transition_21 = pd.DataFrame([])
for status in range(2,30):
    transition_21 = transition_21.append(transition_matrix(1,status))
transition_21 = transition_21.groupby(['product_id','1']).sum().reset_index()
transition_21['post'] = transition_21['positive']/(transition_21['positive'] + transition_21['negative'])
transition_21['supp'] = transition_21['positive'] + transition_21['negative']
transition_21 = transition_21.merge(priors, on='product_id', how='inner')
transition_21['mc1'] = transition_21.apply(lambda x : barreca(x['post'], x['prd_post'], x['supp']), axis=1)
transition_21 = transition_21[['product_id','1','mc1']] 

transition_22 = pd.DataFrame([])
for status in range(2,30):
    transition_22 = transition_22.append(transition_matrix(2,status))
transition_22 = transition_22.groupby(['product_id','1','2']).sum().reset_index()
transition_22['post'] = transition_22['positive']/(transition_22['positive'] + transition_22['negative'])
transition_22['supp'] = transition_22['positive'] + transition_22['negative']
transition_22 = transition_22.merge(priors, on='product_id', how='inner')
transition_22['mc2'] = transition_22.apply(lambda x : barreca(x['post'], x['prd_post'], x['supp']), axis=1)
transition_22 = transition_22[['product_id','1','2','mc2']]

transition_23 = pd.DataFrame([])
for status in range(2,30):
    transition_23 = transition_23.append(transition_matrix(3,status))
transition_23 = transition_23.groupby(['product_id','1','2','3']).sum().reset_index()
transition_23['post'] = transition_23['positive']/(transition_23['positive'] + transition_23['negative'])
transition_23['supp'] = transition_23['positive'] + transition_23['negative']
transition_23 = transition_23.merge(priors, on='product_id', how='inner')
transition_23['mc3'] = transition_23.apply(lambda x : barreca(x['post'], x['prd_post'], x['supp']), axis=1)
transition_23 = transition_23[['product_id','1','2','3','mc3']]

transition_24 = pd.DataFrame([])
for status in range(2,30):
    transition_24 = transition_24.append(transition_matrix(4,status))
transition_24 = transition_24.groupby(['product_id','1','2','3','4']).sum().reset_index()
transition_24['post'] = transition_24['positive']/(transition_24['positive'] + transition_24['negative'])
transition_24['supp'] = transition_24['positive'] + transition_24['negative']
transition_24 = transition_24.merge(priors, on='product_id', how='inner')
transition_24['mc4'] = transition_24.apply(lambda x : barreca(x['post'], x['prd_post'], x['supp']), axis=1)
transition_24 = transition_24[['product_id','1','2','3','4','mc4']]

transition_25 = pd.DataFrame([])
for status in range(2,30):
    transition_25 = transition_25.append(transition_matrix(5,status))
transition_25 = transition_25.groupby(['product_id','1','2','3','4','5']).sum().reset_index()
transition_25['post'] = transition_25['positive']/(transition_25['positive'] + transition_25['negative'])
transition_25['supp'] = transition_25['positive'] + transition_25['negative']
transition_25 = transition_25.merge(priors, on='product_id', how='inner')
transition_25['mc5'] = transition_25.apply(lambda x : barreca(x['post'], x['prd_post'], x['supp']), axis=1)
transition_25 = transition_25[['product_id','1','2','3','4','5','mc5']]

transition = transition_25.merge(transition_24, on=['product_id','1','2','3','4'], how='outer')
transition = transition.merge(transition_23, on=['product_id','1','2','3'], how='outer')
transition = transition.merge(transition_22, on=['product_id','1','2'], how='outer')
transition = transition.merge(transition_21, on=['product_id','1'], how='outer')
transition['1'] = transition['1'].fillna(0) 
transition['2'] = transition['2'].fillna(0)
transition['3'] = transition['3'].fillna(0) 
transition['4'] = transition['4'].fillna(0) 
transition['5'] = transition['5'].fillna(0) 
transition.to_csv('../data/similarity/transition_matrix.csv', index=False)