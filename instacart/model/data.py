import pandas as pd
import numpy as np

dependent = pd.read_csv('../data/model/dependent/dependent_n.csv')[['user_id','product_id']]
print(dependent.shape)

def merge(file, key):
    data = pd.read_csv(file)
    global dependent
    dependent = dependent.merge(data, on=key, how='left')
    print(data.shape, dependent.shape)
    del data
    return None

merge('../data/gensim/cluster_user.csv', ['user_id'])
merge('../data/gensim/xgb_w2_score_n.csv', ['user_id','product_id'])
merge('../data/ffm/ffm_score.csv', ['user_id','product_id'])
merge('../data/similarity/similarity_score.csv', ['user_id','product_id'])
merge('../data/catboost/catboost.csv', ['user_id','product_id'])
merge('../data/catboost/catboost1.csv', ['user_id','product_id'])
merge('../data/profile/product_basic_profile.csv', ['product_id'])
merge('../data/profile/product_brrc_profile.csv', ['product_id'])
merge('../data/profile/product_time_profile.csv', ['product_id'])
merge('../data/profile/user_profile.csv', ['user_id'])
merge('../data/profile/user_brrc_profile.csv', ['user_id'])
merge('../data/profile/user_fscore_profile.csv', ['user_id'])
merge('../data/profile/user_product_profile.csv', ['user_id','product_id'])
merge('../data/profile/user_product_transition.csv', ['user_id','product_id'])
merge('../data/profile/user_product_time.csv', ['user_id','product_id'])
merge('../data/profile/user_product_curr_seq.csv', ['user_id','product_id'])
merge('../data/profile/user_product_max_seq.csv', ['user_id','product_id'])
merge('../data/profile/interaction_profile.csv', ['user_id','product_id'])
merge('../data/profile/user_2way.csv', ['user_id','product_id'])
merge('../data/profile/product_2way.csv', ['user_id','product_id'])

orders = pd.read_csv('../data/driver/driver_order.csv')
orders = orders[orders['counter'] == 1]
orders = orders[['user_id','order_number','order_dow','order_hour_of_day','days_since_prior_order']]

dependent = dependent.merge(orders, on='user_id', how='left')
print(orders.shape, dependent.shape)

dependent['ratio1'] = dependent['usr_prd_lstdy'] / dependent['usr_ais_lstdy'].clip_lower(1)
dependent['ratio2'] = dependent['usr_prd_lstdy'] / dependent['days_since_prior_order'].clip_lower(1)
dependent['ratio3'] = dependent['usr_prd_lstdy'] / dependent['usr_prd_lst_ord'].clip_lower(1)
dependent['ratio4'] = dependent['usr_prd_lstdy'] / dependent['prd_med_dydiff'].clip_lower(1)
dependent['ratio5'] = dependent['usr_sum_rdr'] / dependent['usr_cds_prd']
dependent['ratio6'] = dependent['days_since_prior_order'] / dependent['usr_med_dysc'].clip_lower(1)
dependent['ratio7'] = dependent['prd_post'] * dependent['usr_post']
dependent['ratio8'] = dependent['prd_med_addcrt'] / dependent['cartlen']
dependent['ratio9'] = dependent['usr_prd_adct']  / dependent['cartlen']
dependent['ratio10'] = dependent['cartdiv']  / dependent['cartlen']
dependent['ratio11'] = dependent['usr_prd_lst_ord'] / dependent['usr_cnt_ord']
dependent.to_csv('../data/model/independent/independent_n.csv', index=False)
