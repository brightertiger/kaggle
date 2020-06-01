import pandas as pd 
import numpy as np 
import feather
from functools import reduce
from numpy.linalg import norm
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool 

def embed_matrix(model_path, save_path):
    valid_data = []
    new_whale_data = []
    for idx in tqdm(range(5)):
        temp = pd.read_csv(model_path + '/fold-{}/valid.csv'.format(idx))
        valid_data.append(temp)
        temp = pd.read_csv(model_path + '/fold-{}/new_whale.csv'.format(idx))
        new_whale_data.append(temp)
    valid_data = reduce(lambda x,y : x.append(y), valid_data)
    new_whale_data = reduce(lambda x,y : x.append(y), new_whale_data)
    valid_data = valid_data.groupby('Image', as_index=False).mean()
    new_whale_data = new_whale_data.groupby('Image', as_index=False).mean()
    embed_data = valid_data.append(new_whale_data).reset_index(drop=True)
    del valid_data, new_whale_data
    print('Embedding Data:', embed_data.shape)
    embed_data.to_feather(save_path + 'embedding.feather')
    embed_data = embed_data.set_index('Image')
    return None

def positive_pairs(label_path, matrix_path, save_path):

    embed_data = feather.read_dataframe(matrix_path + 'embedding.feather').set_index('Image')

    def call_similarity(image_1, image_2):
        image_1 = np.array(embed_data.loc[image_1,:]).reshape(-1,)
        image_2 = np.array(embed_data.loc[image_2,:]).reshape(-1,)
        distance = np.dot(image_1, image_2) / (norm(image_1) * norm(image_2))
        return distance
    similarity = lambda x : call_similarity(x[0],x[1])

    encoder = LabelEncoder()
    positive_pairs = pd.read_csv(label_path + 'train.csv')
    positive_pairs = positive_pairs[positive_pairs['Id'] != 'new_whale']
    positive_pairs['Id'] = encoder.fit_transform(positive_pairs['Id'])
    positive_pairs = positive_pairs.merge(positive_pairs, on='Id')
    positive_pairs.columns = ['anchor','label','positive']
    positive_pairs = positive_pairs[['anchor','positive','label']]
    positive_pairs['class_cnt'] = positive_pairs.groupby('label')['anchor'].transform('count')
    positive_pairs['pos_weight'] = positive_pairs[['anchor','positive']].apply(similarity, axis=1)
    positive_pairs = positive_pairs[positive_pairs['pos_weight'] >= 0.1]
    positive_pairs['pos_rank'] = positive_pairs.groupby(['label'])['pos_weight'].rank(ascending=True)
    positive_pairs = positive_pairs[positive_pairs['pos_rank'] <= 100]
    positive_pairs = positive_pairs.drop('pos_rank', axis=1)
    print('Positive Pairs:', positive_pairs.shape)
    positive_pairs = positive_pairs.reset_index(drop=True)
    positive_pairs.to_feather(save_path + 'positive_pairs.feather')
    return None

def negative_pairs(label_path, positive_path, matrix_path, save_path):

    embed_data = feather.read_dataframe(matrix_path + 'embedding.feather').set_index('Image')

    def call_similarity(image_1, image_2):
        image_1 = np.array(embed_data.loc[image_1,:]).reshape(-1,)
        image_2 = np.array(embed_data.loc[image_2,:]).reshape(-1,)
        distance = np.dot(image_1, image_2) / (norm(image_1) * norm(image_2))
        return distance
    similarity = lambda x : call_similarity(x[0],x[1])

    encoder = LabelEncoder()
    label = pd.read_csv(label_path + 'train.csv')
    new_whale = label[label['Id'] == 'new_whale']
    new_whale['Id'] = -1
    label = label[label['Id'] != 'new_whale']
    label['Id'] = encoder.fit_transform(label['Id'])
    label = label.append(new_whale).reset_index(drop=True)
    label.columns = ['negative','label']

    positive_pairs = feather.read_dataframe(positive_path + 'positive_pairs.feather')
    
    def generate_pairs(idx):
        candidate = label[label['label'] != idx][['negative']].drop_duplicates()
        anchor = positive_pairs[positive_pairs['label'] == idx][['anchor']].drop_duplicates()
        candidate['key'] = 1
        anchor['key'] = 1
        cross_prod = anchor.merge(candidate, on='key').drop('key', axis=1)
        cross_prod['neg_weight'] = cross_prod[['anchor','negative']].apply(similarity, axis=1)
        cross_prod['neg_rank'] = cross_prod.groupby(['anchor'])['neg_weight'].rank(ascending=False)
        basic = cross_prod[cross_prod['neg_rank'] == 1].drop('neg_rank', axis=1)
        remain = cross_prod[cross_prod['neg_rank'] > 1].drop('neg_rank', axis=1)
        add_rows = 100 - basic.shape[0]
        if add_rows > 0:
            remain = remain.sort_values(by='neg_weight', ascending=False)
            remain = remain.reset_index(drop=True)
            remain = remain.iloc[:add_rows,:]
        basic = basic.append(remain).reset_index(drop=True)
        return basic
        
    pool = Pool(processes=15)
    negative_pairs = pool.map(generate_pairs, range(5))
    pool.close()
    negative_pairs = reduce(lambda x,y : x.append(y), negative_pairs)
    negative_pairs = negative_pairs.reset_index(drop=True)
    negative_pairs.to_feather(save_path + 'negative_pairs.feather')

    return None




