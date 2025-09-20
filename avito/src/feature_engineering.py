import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from stop_words import get_stop_words
from typing import List, Dict, Any, Tuple
import os


class TextPreprocessor:
    def __init__(self):
        self.russian_vowels = ['а', 'э', 'ы', 'у', 'о', 'я', 'е', 'ё', 'ю', 'и']
        self.english_chars = 'abcdefghijklmnopqrstuvwxyz' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.stop_words = set(get_stop_words('russian'))
    
    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return 'none'
        
        text = str(text).lower()
        text = " ".join(map(str.strip, re.split(r'(\d+)', text)))
        regex = re.compile(r'[^[:alpha:]]')
        text = regex.sub(" ", text)
        text = re.sub('[' + string.punctuation + ']', ' ', text)
        text = " ".join(text.split())
        return text
    
    def has_russian_vowels(self, text: str) -> int:
        for char in text:
            if char in self.russian_vowels:
                return 1
        return 0
    
    def count_english_chars(self, text: str) -> int:
        count = 0
        for char in text:
            if char in self.english_chars:
                count += 1
        return count
    
    def count_punctuation(self, text: str) -> int:
        count = 0
        for char in text:
            if not str.isalnum(char):
                count += 1
        return count
    
    def count_words(self, text: str) -> int:
        return len(text.split())
    
    def average_word_length(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0
        total_length = sum(len(word) for word in words)
        return total_length / len(words)
    
    def count_numeric_chars(self, text: str) -> int:
        count = 0
        for char in text:
            if str.isnumeric(char):
                count += 1
        return count
    
    def uppercase_ratio(self, text: str) -> float:
        if not text:
            return 0
        total_chars = len([c for c in text if c != ' '])
        if total_chars == 0:
            return 0
        
        uppercase_chars = sum(1 for c in text if c == c.upper() and c != ' ')
        return uppercase_chars / total_chars
    
    def all_caps_ratio(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0
        
        all_caps_words = sum(1 for word in words if word == word.upper())
        return all_caps_words / len(words)
    
    def count_stop_words(self, text: str) -> int:
        words = text.lower().split()
        return sum(1 for word in words if word in self.stop_words)


class CountFeatures:
    def __init__(self, config):
        self.config = config
        self.data_loader = None
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
    
    def generate_count_features(self):
        columns = self.config.avito.CATEGORICAL_COLUMNS + ['image_top_1']
        
        train_data = self.data_loader.load_train_data(columns + [self.config.avito.ID_COLUMN])
        test_data = self.data_loader.load_test_data(columns + [self.config.avito.ID_COLUMN])
        full_data = pd.concat([train_data, test_data], ignore_index=True)
        
        train_other = self.data_loader.load_train_active(columns)
        test_other = self.data_loader.load_test_active(columns)
        other_data = pd.concat([train_other, test_other], ignore_index=True)
        
        count_features = [
            (['category_name', 'city'], 'count_1'),
            (['category_name'], 'count_2'),
            (['user_id'], 'count_3'),
            (['user_id', 'category_name'], 'count_4'),
            (['image_top_1'], 'count_5'),
            (['param_1'], 'count_6'),
            (['param_1', 'param_2'], 'count_7'),
            (['param_1', 'param_2', 'param_3'], 'count_8')
        ]
        
        for group_cols, feature_name in count_features:
            feature = other_data.groupby(group_cols)['user_type'].count().reset_index()
            feature = feature.rename(columns={'user_type': feature_name})
            
            driver_cols = [self.config.avito.ID_COLUMN] + group_cols
            driver = full_data[driver_cols].copy()
            driver = driver.merge(feature, on=group_cols, how='left')
            driver = driver[[self.config.avito.ID_COLUMN, feature_name]]
            
            output_path = f'{self.config.avito.FEATURES_DIR}/count/{feature_name}.csv'
            driver.to_csv(output_path, index=False)
            print(f'Generated {feature_name}: {driver.shape}')


class TextFeatures:
    def __init__(self, config):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.data_loader = None
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
    
    def generate_title_features(self):
        train_data = self.data_loader.load_train_data(['title'])
        test_data = self.data_loader.load_test_data(['title'])
        
        train_data[self.config.avito.ID_COLUMN] = self.data_loader.load_train_data([self.config.avito.ID_COLUMN])[self.config.avito.ID_COLUMN]
        test_data[self.config.avito.ID_COLUMN] = self.data_loader.load_test_data([self.config.avito.ID_COLUMN])[self.config.avito.ID_COLUMN]
        
        train_data['title'] = train_data['title'].fillna(' ')
        test_data['title'] = test_data['title'].fillna(' ')
        
        features = {
            'txt_tl_1': lambda x: self.preprocessor.has_russian_vowels(x),
            'txt_tl_2': lambda x: self.preprocessor.count_english_chars(x),
            'txt_tl_3': lambda x: self.preprocessor.count_punctuation(x),
            'txt_tl_4': lambda x: self.preprocessor.count_words(x),
            'txt_tl_5': lambda x: self.preprocessor.average_word_length(x),
            'txt_tl_6': lambda x: self.preprocessor.count_numeric_chars(x),
            'txt_tl_7': lambda x: len(x),
            'txt_tl_8': lambda x: self.preprocessor.uppercase_ratio(x),
            'txt_tl_9': lambda x: self.preprocessor.all_caps_ratio(x),
            'txt_tl_10': lambda x: self.preprocessor.count_stop_words(x)
        }
        
        for feature_name, func in features.items():
            train_data[feature_name] = train_data['title'].map(func)
            test_data[feature_name] = test_data['title'].map(func)
        
        self._generate_sentiment_features(train_data, test_data)
        
        final_features = train_data.append(test_data).drop('title', axis=1)
        output_path = f'{self.config.avito.FEATURES_DIR}/text_title/title.csv'
        final_features.to_csv(output_path, index=False)
        print(f'Generated title features: {final_features.shape}')
    
    def _generate_sentiment_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        train_actuals = self.data_loader.load_train_data([self.config.avito.TARGET_COLUMN])
        
        vocab = list(train_data['title'].values) + list(test_data['title'].values)
        vocab = ' '.join(vocab).lower().split()
        vocab = Counter(vocab)
        vocab = [word for word, count in vocab.items() if count > 1000 and word not in self.preprocessor.stop_words]
        
        positive_words = []
        negative_words = []
        
        for word in vocab:
            dummy = train_data['title'].map(lambda x: 1 if word in x.lower() else 0)
            positive = train_actuals[dummy == 1][self.config.avito.TARGET_COLUMN].mean()
            negative = train_actuals[dummy == 0][self.config.avito.TARGET_COLUMN].mean()
            
            if positive > negative * 1.5:
                positive_words.append(word)
            elif negative > positive * 1.5:
                negative_words.append(word)
        
        def count_positive_words(text):
            words = text.split()
            return len([w for w in words if w.lower() in positive_words])
        
        def count_negative_words(text):
            words = text.split()
            return len([w for w in words if w.lower() in negative_words])
        
        train_data['txt_tl_pos'] = train_data['title'].map(count_positive_words)
        test_data['txt_tl_pos'] = test_data['title'].map(count_positive_words)
        train_data['txt_tl_neg'] = train_data['title'].map(count_negative_words)
        test_data['txt_tl_neg'] = test_data['title'].map(count_negative_words)


class UserFeatures:
    def __init__(self, config):
        self.config = config
        self.data_loader = None
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
    
    def generate_user_features(self):
        columns = [self.config.avito.ID_COLUMN, 'user_id', 'param_1', 'region', 'title',
                  'parent_category_name', 'param_2', 'param_3', 'city', 'category_name']
        
        train_data = self.data_loader.load_train_data(columns)
        test_data = self.data_loader.load_test_data(columns)
        source_1 = pd.concat([train_data, test_data], ignore_index=True)
        
        train_other = self.data_loader.load_train_active(columns)
        test_other = self.data_loader.load_test_active(columns)
        source_2 = pd.concat([train_other, test_other], ignore_index=True)
        
        data = pd.concat([source_1, source_2], ignore_index=True)
        
        user_features = [
            (['user_id'], 'title', 'nunique', 'user_1'),
            (['user_id'], 'category_name', 'nunique', 'user_2'),
            (['user_id'], 'param_1', 'nunique', 'user_3'),
        ]
        
        features_data = []
        for group_cols, agg_col, agg_func, feature_name in user_features:
            feature = data.groupby(group_cols)[agg_col].agg(agg_func).reset_index()
            feature = feature.rename(columns={agg_col: feature_name})
            
            driver_cols = [self.config.avito.ID_COLUMN] + group_cols
            driver = source_1[driver_cols].copy()
            driver = driver.merge(feature, on=group_cols, how='left').fillna(0)
            driver = driver[[self.config.avito.ID_COLUMN, feature_name]]
            features_data.append(driver)
        
        null_feature = data.copy()
        null_feature['null'] = null_feature.isnull().sum(axis=1)
        null_feature = null_feature.groupby('user_id')['null'].sum().reset_index()
        null_feature = null_feature.rename(columns={'null': 'user_4'})
        
        driver = source_1[[self.config.avito.ID_COLUMN, 'user_id']].copy()
        driver = driver.merge(null_feature, on='user_id', how='left').fillna(0)
        driver = driver[[self.config.avito.ID_COLUMN, 'user_4']]
        features_data.append(driver)
        
        matrix = features_data[0]
        for feature_df in features_data[1:]:
            matrix = matrix.merge(feature_df, on=self.config.avito.ID_COLUMN)
        
        for col in ['user_1', 'user_2', 'user_3', 'user_4']:
            matrix[col] = self._normalize_feature(matrix[col])
        
        output_path = f'{self.config.avito.FEATURES_DIR}/user/user_features.csv'
        matrix.to_csv(output_path, index=False)
        print(f'Generated user features: {matrix.shape}')
    
    def _normalize_feature(self, feature: pd.Series) -> pd.Series:
        upper_bound = feature.quantile(0.70)
        feature = feature.clip(upper=upper_bound)
        mean = feature.mean()
        std = feature.std()
        return (feature - mean) / std


class DateFeatures:
    def __init__(self, config):
        self.config = config
        self.data_loader = None
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
    
    def generate_date_features(self):
        columns = [self.config.avito.ID_COLUMN, 'activation_date']
        
        train_data = self.data_loader.load_train_data(columns)
        test_data = self.data_loader.load_test_data(columns)
        full_data = pd.concat([train_data, test_data], ignore_index=True)
        
        full_data['activation_date'] = pd.to_datetime(full_data['activation_date'])
        full_data['week_day'] = full_data['activation_date'].dt.weekday
        full_data = full_data.drop(['activation_date'], axis=1)
        
        output_path = f'{self.config.avito.FEATURES_DIR}/date/time.csv'
        full_data.to_csv(output_path, index=False)
        print(f'Generated date features: {full_data.shape}')


class FeaturePipeline:
    def __init__(self, config):
        self.config = config
        self.data_loader = None
        
        self.count_features = CountFeatures(config)
        self.text_features = TextFeatures(config)
        self.user_features = UserFeatures(config)
        self.date_features = DateFeatures(config)
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
        self.count_features.set_data_loader(data_loader)
        self.text_features.set_data_loader(data_loader)
        self.user_features.set_data_loader(data_loader)
        self.date_features.set_data_loader(data_loader)
    
    def generate_all_features(self):
        print("Generating count features...")
        self.count_features.generate_count_features()
        
        print("Generating text features...")
        self.text_features.generate_title_features()
        
        print("Generating user features...")
        self.user_features.generate_user_features()
        
        print("Generating date features...")
        self.date_features.generate_date_features()
        
        print("All features generated successfully!")
