
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import lightgbm as lgb


# In[2]:

score_data = pd.read_feather('../data/data/model/score_data.feather', nthreads=15)
score_driver = score_data[['click_id']].copy()
score_data = score_data.drop(['day','click_id'], axis=1)


# In[3]:

model = lgb.Booster(model_file='../data/data/model/lightgbm_2.model')


# In[4]:

score_driver['is_attributed'] = model.predict(score_data)


# In[5]:

score_driver.to_csv('../data/full/lightgbm_6.csv', index=False)


# In[ ]:



