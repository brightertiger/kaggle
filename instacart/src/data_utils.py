#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import gensim
from gensim.models import Word2Vec

from .config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.ensure_directories()
    
    def ensure_directories(self):
        os.makedirs(self.config.get_data_path(), exist_ok=True)
        os.makedirs(self.config.get_data_path('download'), exist_ok=True)
        os.makedirs(self.config.get_data_path('driver'), exist_ok=True)
        os.makedirs(self.config.get_data_path('profile'), exist_ok=True)
        os.makedirs(self.config.get_data_path('model'), exist_ok=True)
        os.makedirs(self.config.get_data_path('model', 'dependent'), exist_ok=True)
        os.makedirs(self.config.get_data_path('model', 'independent'), exist_ok=True)
        os.makedirs(self.config.get_output_path(), exist_ok=True)
        os.makedirs(self.config.get_model_path(), exist_ok=True)
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        orders_path = self.config.get_data_path('download', 'orders.csv')
        products_path = self.config.get_data_path('download', 'products.csv')
        order_products_prior_path = self.config.get_data_path('download', 'order_products__prior.csv')
        order_products_train_path = self.config.get_data_path('download', 'order_products__train.csv')
        
        orders = pd.read_csv(orders_path)
        products = pd.read_csv(products_path)
        order_products_prior = pd.read_csv(order_products_prior_path)
        order_products_train = pd.read_csv(order_products_train_path)
        
        return orders, products, order_products_prior, order_products_train
    
    def create_user_splits(self, orders: pd.DataFrame) -> pd.DataFrame:
        def holdout(user_id, eval_set):
            if eval_set == 'train':
                if user_id % 10 >= 8:
                    return 'valid'
                else:
                    return eval_set
            else:
                return eval_set
        
        users = orders[['user_id', 'eval_set']].copy()
        users = users[users['eval_set'] != 'prior']
        users = users.drop_duplicates()
        users['eval_set'] = users.apply(lambda x: holdout(x['user_id'], x['eval_set']), axis=1)
        
        return users
    
    def create_order_features(self, orders: pd.DataFrame) -> pd.DataFrame:
        orders_sorted = orders.sort_values(['user_id', 'order_number'], ascending=[True, False])
        orders_sorted['counter'] = orders_sorted.groupby('user_id')['order_number'].rank(ascending=False)
        orders_sorted = orders_sorted.sort_values(by=['user_id', 'counter'], ascending=[True, True])
        
        return orders_sorted
    
    def create_product_features(self, products: pd.DataFrame) -> pd.DataFrame:
        products_clean = products.copy()
        products_clean = products_clean.append(pd.DataFrame([[0, -1, -1]], columns=products_clean.columns))
        products_clean = products_clean.sort_values(by=['product_id'])
        
        return products_clean
    
    def create_order_products(self, order_products_prior: pd.DataFrame, 
                            order_products_train: pd.DataFrame, 
                            orders: pd.DataFrame) -> pd.DataFrame:
        order_prods = order_products_prior.append(order_products_train)
        
        none_candidate = orders[orders['order_number'] > 1][['order_id']]
        none_orders = order_prods.groupby('order_id')['reordered'].max().reset_index()
        none_orders = none_orders[none_orders['reordered'] == 0][['order_id']]
        none_orders = none_orders.merge(none_candidate, on='order_id', how='inner')
        none_orders['product_id'] = [0] * none_orders.shape[0]
        none_orders['add_to_cart_order'] = [0] * none_orders.shape[0]
        none_orders['reordered'] = [1] * none_orders.shape[0]
        
        order_prods = order_prods.append(none_orders).sort_values(by=['order_id', 'add_to_cart_order'])
        order_prods = order_prods.reset_index(drop=True)
        
        return order_prods
    
    def save_processed_data(self, users: pd.DataFrame, orders: pd.DataFrame, 
                          products: pd.DataFrame, order_products: pd.DataFrame):
        users.to_csv(self.config.get_data_path('driver', 'driver_user.csv'), index=False)
        orders.to_csv(self.config.get_data_path('driver', 'driver_order.csv'), index=False)
        products.to_csv(self.config.get_data_path('driver', 'driver_product.csv'), index=False)
        order_products.to_csv(self.config.get_data_path('driver', 'driver_order_products.csv'), index=False)


class FeatureProcessor:
    def __init__(self, config: Config):
        self.config = config
    
    def clean_product_name(self, name: str) -> str:
        name = name.replace('-', ' ')
        name = name.replace('&', ' ')
        name = name.replace("'N", ' ')
        string = name.split()
        string = [''.join(filter(str.isalpha, x.lower())) for x in string]
        return ' '.join(string)
    
    def create_tfidf_features(self, product_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        clean_names = [self.clean_product_name(name) for name in product_names]
        
        vectorizer = CountVectorizer(max_features=self.config.feature_params['tfidf_max_features'])
        tfidf = TfidfTransformer(norm="l2", smooth_idf=True)
        
        vec_name = vectorizer.fit_transform(clean_names)
        vec_idf = tfidf.fit_transform(vec_name)
        vec_idf = vec_idf.todense()
        
        mean_tfidf = vec_idf.sum(axis=1) / np.array([len(name) for name in clean_names]).reshape(-1, 1)
        max_tfidf = vec_idf.max(axis=1)
        
        return mean_tfidf, max_tfidf
    
    def create_word2vec_model(self, product_names: List[str]) -> gensim.models.Word2Vec:
        clean_names = [self.clean_product_name(name).split() for name in product_names]
        
        model = Word2Vec(
            clean_names,
            vector_size=self.config.feature_params['word2vec_dim'],
            window=self.config.feature_params['word2vec_window'],
            min_count=self.config.feature_params['word2vec_min_count'],
            workers=4,
            seed=self.config.random_seed
        )
        
        return model
    
    def create_product_embeddings(self, model: gensim.models.Word2Vec, 
                                product_names: List[str]) -> np.ndarray:
        embeddings = []
        for name in product_names:
            words = self.clean_product_name(name).split()
            word_vectors = [model.wv[word] for word in words if word in model.wv]
            
            if word_vectors:
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.config.feature_params['word2vec_dim']))
        
        return np.array(embeddings)


class TargetCreator:
    def __init__(self, config: Config):
        self.config = config
    
    def create_target_variable(self, cutoff: int, products: pd.DataFrame, 
                             orders: pd.DataFrame) -> pd.DataFrame:
        candidate = products[products['counter'] > cutoff][['user_id', 'product_id', 'aisle_id', 'eval_set']]
        nones = candidate[['user_id', 'eval_set']].drop_duplicates()
        nones['product_id'] = [0] * nones.shape[0]
        candidate = candidate.append(nones).drop_duplicates()
        
        dependent = products[products['counter'] == cutoff][['user_id', 'product_id', 'reordered']]
        user_list = list(set(dependent['user_id']))
        
        data = candidate.merge(dependent, on=['user_id', 'product_id'], how='left', indicator=True)
        data = data[data['user_id'].isin(user_list)]
        data['reordered'] = data['reordered'].fillna(0)
        
        return data[['user_id', 'product_id', 'aisle_id', 'eval_set', 'reordered']]
    
    def create_dependent_datasets(self, products: pd.DataFrame, orders: pd.DataFrame):
        for n in [1, 2]:
            target_data = self.create_target_variable(n, products, orders)
            target_data.to_csv(
                self.config.get_data_path('model', 'dependent', f'dependent_n_{n}.csv'), 
                index=False
            )
        
        target_data = self.create_target_variable(0, products, orders)
        target_data.to_csv(
            self.config.get_data_path('model', 'dependent', 'dependent_n.csv'), 
            index=False
        )


def create_train_valid_split(data: pd.DataFrame, target_col: str = 'reordered', 
                           test_size: float = 0.2, random_state: int = 108) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, valid_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=data[target_col]
    )
    return train_data, valid_data


def calculate_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return 2 * auc(fpr, tpr) - 1
