import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

train_data = pd.read_json('../data/download/train.json')
train_data['inc_angle'] = pd.to_numeric(train_data['inc_angle'], errors='coerce')
print('train data:', train_data.shape)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(2,2,1)
arr = np.reshape(np.array(train_data.iloc[0,0]),(75,75))
ax.imshow(arr,cmap='jet')
ax = fig.add_subplot(2,2,2)
arr = np.reshape(np.array(train_data.iloc[0,1]),(75,75))
ax.imshow(arr,cmap='jet')
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(2,2,1)
arr = np.reshape(np.array(train_data.iloc[2,0]),(75,75))
ax.imshow(arr,cmap='jet')
ax = fig.add_subplot(2,2,2)
arr = np.reshape(np.array(train_data.iloc[2,1]),(75,75))
ax.imshow(arr,cmap='jet')
plt.show()