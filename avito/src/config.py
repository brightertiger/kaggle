import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class AvitoConfig:
    RANDOM_STATE: int = 2017
    N_FOLDS: int = 5
    
    DATA_ROOT: str = '../../data'
    TRAIN_DATA_PATH: str = '../../data/download/train.csv'
    TEST_DATA_PATH: str = '../../data/download/test.csv'
    TRAIN_ACTIVE_PATH: str = '../../data/download/train_active.csv'
    TEST_ACTIVE_PATH: str = '../../data/download/test_active.csv'
    
    FEATURES_DIR: str = '../../data/data/features'
    MODEL_DIR: str = '../../model'
    OUTPUT_DIR: str = '../../output'
    
    TARGET_COLUMN: str = 'deal_probability'
    ID_COLUMN: str = 'item_id'
    
    TEXT_COLUMNS: List[str] = None
    CATEGORICAL_COLUMNS: List[str] = None
    NUMERICAL_COLUMNS: List[str] = None
    
    def __post_init__(self):
        if self.TEXT_COLUMNS is None:
            self.TEXT_COLUMNS = ['title', 'description']
        
        if self.CATEGORICAL_COLUMNS is None:
            self.CATEGORICAL_COLUMNS = [
                'parent_category_name', 'category_name', 'user_type',
                'region', 'city', 'user_id', 'param_1', 'param_2', 'param_3'
            ]
        
        if self.NUMERICAL_COLUMNS is None:
            self.NUMERICAL_COLUMNS = [
                'price', 'image_top_1', 'activation_date'
            ]
    
    def create_directories(self):
        os.makedirs(self.FEATURES_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        feature_subdirs = [
            'count', 'date', 'duplicate', 'encode', 'item', 'relative',
            'renew', 'svd', 'text_desc', 'text_title', 'user'
        ]
        
        for subdir in feature_subdirs:
            os.makedirs(os.path.join(self.FEATURES_DIR, subdir), exist_ok=True)


@dataclass
class ModelConfig:
    RIDGE_ALPHA: float = 20.0
    RIDGE_MAX_ITER: int = None
    RIDGE_TOL: float = 0.001
    RIDGE_SOLVER: str = 'auto'
    
    TFIDF_MAX_FEATURES: int = 50000
    TFIDF_NGRAM_RANGE: tuple = (1, 2)
    TFIDF_SUBLINEAR_TF: bool = True
    TFIDF_NORM: str = 'l2'
    TFIDF_SMOOTH_IDF: bool = False
    
    COUNT_VECTORIZER_NGRAM_RANGE: tuple = (1, 2)
    
    ENSEMBLE_WEIGHTS: List[float] = None
    
    def __post_init__(self):
        if self.ENSEMBLE_WEIGHTS is None:
            self.ENSEMBLE_WEIGHTS = [0.5, 0.5]


class Config:
    def __init__(self):
        self.avito = AvitoConfig()
        self.model = ModelConfig()
        self.avito.create_directories()
