import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from .config import Config

class DataLoader:
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Config.DATA_DIR
        
    def load_train_data(self) -> pd.DataFrame:
        train_path = self.data_dir / Config.TRAIN_FILE.name
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        return pd.read_csv(train_path)
    
    def load_test_data(self) -> pd.DataFrame:
        test_path = self.data_dir / Config.TEST_FILE.name
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_path}")
        return pd.read_csv(test_path)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_data = self.load_train_data()
        test_data = self.load_test_data()
        return train_data, test_data
    
    def encode_authors(self, df: pd.DataFrame, author_column: str = 'author') -> pd.DataFrame:
        df = df.copy()
        df[author_column] = df[author_column].map(Config.AUTHOR_MAP)
        return df

class FeatureMerger:
    def __init__(self, score_dir: Path = None):
        self.score_dir = score_dir or Config.SCORE_DIR
        
    def merge_all_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_merged = train_data.copy()
        test_merged = test_data.copy()
        
        try:
            nb_score_train = pd.read_csv(self.score_dir / Config.TRAIN_NB_SCORE).drop(['id', 'author'], axis=1)
            nb_score_test = pd.read_csv(self.score_dir / Config.TEST_NB_SCORE).drop(['id'], axis=1)
            train_merged = train_merged.join(nb_score_train)
            test_merged = test_merged.join(nb_score_test)
        except FileNotFoundError:
            print("Warning: Naive Bayes scores not found")
        
        try:
            nb_feats_train = pd.read_csv(self.score_dir / Config.TRAIN_NB_FEATS)
            nb_feats_test = pd.read_csv(self.score_dir / Config.TEST_NB_FEATS)
            train_merged = train_merged.join(nb_feats_train)
            test_merged = test_merged.join(nb_feats_test)
        except FileNotFoundError:
            print("Warning: Naive Bayes features not found")
        
        try:
            nn_score_train = pd.read_csv(self.score_dir / Config.TRAIN_NN_SCORE)
            nn_score_test = pd.read_csv(self.score_dir / Config.TEST_NN_SCORE)
            
            nn_score_train['nn_prob'] = np.max(
                np.hstack([nn_score_train[['k1']], nn_score_train[['k2']], nn_score_train[['k3']]]), 
                axis=1
            )
            nn_score_train = nn_score_train[['id', 'keras', 'nn_prob']]
            nn_score_test = nn_score_test[['id', 'keras']]
            
            train_merged = train_merged.merge(nn_score_train, on='id')
            test_merged = test_merged.merge(nn_score_test, on='id')
        except FileNotFoundError:
            print("Warning: Neural network scores not found")
        
        try:
            lstm_score_train = pd.read_csv(self.score_dir / Config.TRAIN_LSTM_SCORE)
            lstm_score_test = pd.read_csv(self.score_dir / Config.TEST_LSTM_SCORE)
            
            lstm_score_train['lstm_prob'] = np.max(
                np.hstack([lstm_score_train[['l1']], lstm_score_train[['l2']], lstm_score_train[['l3']]]), 
                axis=1
            )
            lstm_score_train = lstm_score_train[['id', 'lstm', 'lstm_prob']]
            lstm_score_test = lstm_score_test[['id', 'lstm']]
            
            train_merged = train_merged.merge(lstm_score_train, on='id')
            test_merged = test_merged.merge(lstm_score_test, on='id')
        except FileNotFoundError:
            print("Warning: LSTM scores not found")
        
        if 'keras' in train_merged.columns and 'lstm' in train_merged.columns:
            train_merged['agree'] = 1.0 * np.equal(train_merged['keras'], train_merged['lstm'])
        
        return train_merged, test_merged
    
    def prepare_final_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_final = train_data.drop(['text', 'author'], axis=1, errors='ignore')
        test_final = test_data.drop(['text'], axis=1, errors='ignore')
        
        return train_final, test_final

class DataProcessor:
    def __init__(self, data_dir: Path = None, score_dir: Path = None):
        self.data_loader = DataLoader(data_dir)
        self.feature_merger = FeatureMerger(score_dir)
        
    def process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_data, test_data = self.data_loader.load_data()
        
        train_data = self.data_loader.encode_authors(train_data)
        
        train_merged, test_merged = self.feature_merger.merge_all_features(train_data, test_data)
        
        train_final, test_final = self.feature_merger.prepare_final_data(train_merged, test_merged)
        
        return train_final, test_final
