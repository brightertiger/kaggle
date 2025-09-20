#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import gensim
from gensim.models import Word2Vec

from .config import Config
from .data_utils import DataLoader, FeatureProcessor


class UserFeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
        self.processor = FeatureProcessor(config)
    
    def create_basic_features(self, orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        orders_filtered = orders[orders['counter'] > 1]
        products_merged = products.merge(orders_filtered, on='order_id', how='inner')
        products_merged = products_merged.drop(['eval_set'], axis=1)
        
        aggregate = {
            'reordered': np.sum,
            'counter': np.count_nonzero,
            'order_id': pd.Series.nunique,
            'product_id': pd.Series.nunique,
            'aisle_id': pd.Series.nunique,
            'department_id': pd.Series.nunique,
            'days_since_prior_order': np.median
        }
        
        user_common = products_merged.groupby('user_id').agg(aggregate).reset_index()
        features = ['usr_sum_rdr', 'usr_cnt_prd', 'usr_cnt_ord', 'usr_cds_prd',
                   'usr_cds_ais', 'usr_cds_dep', 'usr_med_dysc']
        user_common.columns = ['user_id'] + features
        
        return user_common
    
    def create_cart_features(self, products: pd.DataFrame) -> pd.DataFrame:
        cart_length = products.groupby(['user_id', 'order_id'])['product_id'].count().reset_index()
        cart_length = cart_length.groupby(['user_id'])['product_id'].mean().reset_index()
        cart_length = cart_length.rename(columns={'product_id': 'cartlen'})
        
        cart_diverse = products.groupby(['user_id', 'order_id'])['aisle_id'].apply(pd.Series.nunique).reset_index()
        cart_diverse = cart_diverse.groupby(['user_id'])['aisle_id'].mean().reset_index()
        cart_diverse = cart_diverse.rename(columns={'aisle_id': 'cartdiv'})
        
        avg_reorder = products.groupby(['user_id', 'order_id'])['reordered'].mean().reset_index()
        avg_reorder = avg_reorder.groupby(['user_id'])['reordered'].mean().reset_index()
        avg_reorder = avg_reorder.rename(columns={'reordered': 'usr_avg_rdr'})
        
        return cart_length, cart_diverse, avg_reorder
    
    def create_lag_features(self, dependent_data: pd.DataFrame) -> pd.DataFrame:
        lag_reorder = dependent_data.groupby('user_id')['reordered'].mean().reset_index()
        lag_reorder = lag_reorder.rename(columns={'reordered': 'usr_lag_rdr'})
        return lag_reorder
    
    def build_user_profile(self, users: pd.DataFrame, orders: pd.DataFrame, 
                         products: pd.DataFrame, dependent_data: pd.DataFrame) -> pd.DataFrame:
        user_common = self.create_basic_features(orders, products)
        cart_length, cart_diverse, avg_reorder = self.create_cart_features(products)
        lag_reorder = self.create_lag_features(dependent_data)
        
        user_profile = users.copy()
        user_profile = user_profile.merge(user_common, on='user_id', how='left')
        user_profile = user_profile.merge(cart_length, on='user_id', how='left')
        user_profile = user_profile.merge(cart_diverse, on='user_id', how='left')
        user_profile = user_profile.merge(avg_reorder, on='user_id', how='left')
        user_profile = user_profile.merge(lag_reorder, on='user_id', how='left')
        
        return user_profile


class ProductFeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
        self.processor = FeatureProcessor(config)
    
    def create_common_profile(self, data: pd.DataFrame, level: str, prefix: str) -> pd.DataFrame:
        aggregate = {
            'reordered': [np.sum, np.mean],
            'order_id': pd.Series.nunique,
            'user_id': pd.Series.nunique,
            'order_number': np.median,
            'add_to_cart_order': np.median
        }
        
        grouped_data = data.groupby(level).agg(aggregate).reset_index()
        features = ['sum_rdr', 'avg_rdr', 'cds_ord', 'cds_usr', 'med_ordn', 'med_addcrt']
        grouped_data.columns = [level] + [prefix + x for x in features]
        grouped_data[prefix + 'rt_ord_usr'] = grouped_data[prefix + 'cds_ord'] / grouped_data[prefix + 'cds_usr']
        
        return grouped_data
    
    def create_affinity_features(self, products: pd.DataFrame) -> pd.DataFrame:
        prod_last_order = products.groupby(['user_id', 'product_id'])['order_number'].max().reset_index()
        user_last_order = products.groupby(['user_id'])['order_number'].max().reset_index()
        
        prod_last_order.columns = ['user_id', 'product_id', 'prod_last_order']
        user_last_order.columns = ['user_id', 'user_last_order']
        
        affinity = prod_last_order.merge(user_last_order, on='user_id')
        affinity['order_since'] = affinity['user_last_order'] - affinity['prod_last_order']
        
        prod_affinity = affinity.groupby('product_id')['order_since'].apply(np.median).reset_index()
        prod_affinity.columns = ['product_id', 'product_affinity']
        
        return prod_affinity
    
    def create_text_features(self, product_names: pd.DataFrame) -> pd.DataFrame:
        prod_name = product_names.copy()
        prod_name['length'] = prod_name['product_name'].map(lambda x: len(x))
        prod_name['words'] = prod_name['product_name'].map(lambda x: len(x.split()))
        prod_name['organic'] = prod_name['product_name'].map(lambda x: 1 if 'organic' in x.lower() else 0)
        
        clean_names = list(prod_name['product_name'].map(self.processor.clean_product_name))
        
        vectorizer = CountVectorizer()
        tfidf = TfidfTransformer(norm="l2", smooth_idf=True)
        
        vec_name = vectorizer.fit_transform(clean_names)
        vec_idf = tfidf.fit_transform(vec_name)
        vec_idf = vec_idf.todense()
        
        prod_name['mean_tfidf'] = vec_idf.sum(axis=1) / prod_name['length']
        prod_name['max_tfidf'] = vec_idf.max(axis=1)
        prod_name = prod_name.drop('product_name', axis=1)
        
        return prod_name
    
    def build_product_profile(self, products: pd.DataFrame, product_names: pd.DataFrame) -> pd.DataFrame:
        common_product = self.create_common_profile(products, 'product_id', 'prd_')
        common_department = self.create_common_profile(products, 'department_id', 'dep_')
        common_aisle = self.create_common_profile(products, 'aisle_id', 'ais_')
        
        prod_affinity = self.create_affinity_features(products)
        text_features = self.create_text_features(product_names)
        
        product_profile = products[['product_id', 'aisle_id', 'department_id']].drop_duplicates()
        product_profile = product_profile.merge(common_product, on='product_id', how='left')
        product_profile = product_profile.merge(common_department, on='department_id', how='left')
        product_profile = product_profile.merge(common_aisle, on='aisle_id', how='left')
        product_profile = product_profile.merge(prod_affinity, on='product_id', how='left')
        product_profile = product_profile.merge(text_features, on='product_id', how='left')
        
        product_profile = product_profile.fillna(0.0)
        product_profile = product_profile.drop(['department_id', 'aisle_id'], axis=1)
        
        return product_profile


class MeanEncodingEngineer:
    def __init__(self, config: Config):
        self.config = config
    
    def barreca_encoding(self, posterior: float, n: int, k: int, f: float, 
                        prior: float = None) -> float:
        if prior is None:
            prior = self.config.encoding_params['prior_probability']
        
        factor = np.exp((n - k) / f)
        factor = factor / (factor + 1)
        
        if np.isnan(factor):
            factor = 1.0
        
        return factor * posterior + (1 - factor) * prior
    
    def create_cumulative_features(self, counters: List[int], groupby_cols: List[str], 
                                 products: pd.DataFrame) -> pd.DataFrame:
        cumulative_df = pd.DataFrame()
        
        for counter in counters:
            if counter % 5 == 0:
                print(f"Processing counter {counter}...")
            
            target_data = self._create_target_for_counter(counter, products)
            reorder = target_data.groupby(groupby_cols)['reordered'].sum().reset_index()
            potential = target_data.groupby(groupby_cols)['reordered'].count().reset_index()
            
            reorder = reorder.rename(columns={'reordered': 'sum'})
            potential = potential.rename(columns={'reordered': 'count'})
            
            grouped = reorder.merge(potential, on=groupby_cols, how='inner')
            grouped['counter'] = [counter] * grouped.shape[0]
            cumulative_df = cumulative_df.append(grouped)
        
        return cumulative_df
    
    def _create_target_for_counter(self, cutoff: int, products: pd.DataFrame) -> pd.DataFrame:
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
    
    def create_encoding_features(self, products: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        aisle_cumulative = self.create_cumulative_features(range(2, 30), ['aisle_id'], products)
        aisle_aggregate = aisle_cumulative.groupby(['aisle_id']).agg({'sum': 'sum', 'count': 'sum'}).reset_index()
        aisle_aggregate['ais_post'] = aisle_aggregate['sum'] / aisle_aggregate['count']
        aisle_aggregate = aisle_aggregate.drop(['sum', 'count'], axis=1)
        
        product_cumulative = self.create_cumulative_features(range(2, 30), ['product_id', 'aisle_id'], products)
        product_aggregate = product_cumulative.groupby(['product_id', 'aisle_id']).agg({'sum': 'sum', 'count': 'sum'}).reset_index()
        product_aggregate['prd_post'] = product_aggregate['sum'] / product_aggregate['count']
        product_aggregate = product_aggregate.drop(['sum', 'count'], axis=1)
        product_aggregate = product_aggregate.merge(aisle_aggregate, on='aisle_id', how='inner')
        product_aggregate['prd_ais_post_rt'] = product_aggregate['prd_post'] / product_aggregate['ais_post']
        product_aggregate = product_aggregate.drop(['aisle_id'], axis=1)
        
        user_cumulative = self.create_cumulative_features(range(2, 30), ['user_id'], products)
        user_aggregate = user_cumulative.groupby(['user_id']).agg({'sum': 'sum', 'count': 'sum'}).reset_index()
        user_aggregate['usr_post'] = user_aggregate['sum'] / user_aggregate['count']
        user_aggregate = user_aggregate.drop(['sum', 'count'], axis=1)
        
        return product_aggregate, user_aggregate


class InteractionFeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
    
    def create_user_product_features(self, dependent_data: pd.DataFrame, orders: pd.DataFrame, 
                                   products: pd.DataFrame) -> pd.DataFrame:
        target = orders[orders['counter'] == 1]
        user_product_profile = products[['user_id', 'product_id', 'usr_prd_cnt']]
        history = products.merge(orders, on='order_id', how='inner')
        
        target = dependent_data.merge(target, on=['user_id'], how='inner')
        target = target[['user_id', 'product_id', 'order_dow', 'order_hour_of_day', 
                        'days_since_prior_order', 'order_number']]
        target = target.merge(user_product_profile, on=['user_id', 'product_id'])
        
        aggregate = {'order_id': 'count'}
        
        usr_prd_dow = history.groupby(['user_id', 'product_id', 'order_dow']).agg(aggregate).reset_index()
        usr_prd_dow = usr_prd_dow.rename(columns={'order_id': 'usr_prd_dow_cnt'})
        
        history['order_hour_of_day'], hour_bins = pd.qcut(history['order_hour_of_day'], 10, 
                                                         retbins=True, labels=False)
        usr_prd_hod = history.groupby(['user_id', 'product_id', 'order_hour_of_day']).agg(aggregate).reset_index()
        usr_prd_hod = usr_prd_hod.rename(columns={'order_id': 'usr_prd_hod_cnt'})
        
        history['days_since_prior_order'], days_bins = pd.qcut(history['days_since_prior_order'], 7, 
                                                              retbins=True, labels=False)
        usr_prd_dysc = history.groupby(['user_id', 'product_id', 'days_since_prior_order']).agg(aggregate).reset_index()
        usr_prd_dysc = usr_prd_dysc.rename(columns={'order_id': 'usr_prd_dysc_cnt'})
        
        target['order_hour_of_day'] = pd.cut(target['order_hour_of_day'], bins=hour_bins, 
                                           labels=False, include_lowest=True)
        target['days_since_prior_order'] = pd.cut(target['days_since_prior_order'], bins=days_bins, 
                                                labels=False, include_lowest=True)
        
        target = target.merge(usr_prd_dow, on=['user_id', 'product_id', 'order_dow'], how='left')
        target = target.merge(usr_prd_hod, on=['user_id', 'product_id', 'order_hour_of_day'], how='left')
        target = target.merge(usr_prd_dysc, on=['user_id', 'product_id', 'days_since_prior_order'], how='left')
        
        target = target.drop(['order_dow', 'order_hour_of_day', 'days_since_prior_order', 'order_number'], axis=1)
        
        target['usrprd2way1'] = target['usr_prd_dow_cnt'] / target['usr_prd_cnt']
        target['usrprd2way2'] = target['usr_prd_hod_cnt'] / target['usr_prd_cnt']
        target['usrprd2way3'] = target['usr_prd_dysc_cnt'] / target['usr_prd_cnt']
        target = target.drop(['usr_prd_cnt'], axis=1)
        
        return target


class Word2VecEngineer:
    def __init__(self, config: Config):
        self.config = config
    
    def create_product_embeddings(self, products: pd.DataFrame, product_names: pd.DataFrame) -> Tuple[Word2Vec, pd.DataFrame]:
        clean_names = []
        for name in product_names['product_name']:
            clean_name = name.replace('-', ' ').replace('&', ' ').replace("'N", ' ')
            words = [''.join(filter(str.isalpha, x.lower())) for x in name.split()]
            clean_names.append(words)
        
        model = Word2Vec(
            clean_names,
            vector_size=self.config.feature_params['word2vec_dim'],
            window=self.config.feature_params['word2vec_window'],
            min_count=self.config.feature_params['word2vec_min_count'],
            workers=4,
            seed=self.config.random_seed
        )
        
        embeddings = []
        for name in product_names['product_name']:
            clean_name = name.replace('-', ' ').replace('&', ' ').replace("'N", ' ')
            words = [''.join(filter(str.isalpha, x.lower())) for x in name.split()]
            
            word_vectors = [model.wv[word] for word in words if word in model.wv]
            if word_vectors:
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.config.feature_params['word2vec_dim']))
        
        embedding_df = pd.DataFrame(embeddings)
        embedding_df.columns = [f'prod_vec_{i}' for i in range(embedding_df.shape[1])]
        embedding_df['product_id'] = product_names['product_id'].values
        
        return model, embedding_df
    
    def create_user_embeddings(self, products: pd.DataFrame, product_model: Word2Vec) -> pd.DataFrame:
        user_products = products.groupby('user_id')['product_id'].apply(list).reset_index()
        
        user_embeddings = []
        for user_id, product_list in user_products.values:
            product_vectors = []
            for prod_id in product_list:
                if prod_id in product_model.wv:
                    product_vectors.append(product_model.wv[prod_id])
            
            if product_vectors:
                user_embedding = np.mean(product_vectors, axis=0)
            else:
                user_embedding = np.zeros(self.config.feature_params['word2vec_dim'])
            
            user_embeddings.append(user_embedding)
        
        embedding_df = pd.DataFrame(user_embeddings)
        embedding_df.columns = [f'user_vec_{i}' for i in range(embedding_df.shape[1])]
        embedding_df['user_id'] = user_products['user_id'].values
        
        return embedding_df
