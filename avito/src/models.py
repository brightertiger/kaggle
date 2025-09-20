import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import mean_squared_error
from typing import List, Dict, Any, Tuple
import os


class TextModel:
    def __init__(self, config):
        self.config = config
        self.data_loader = None
        self.preprocessor = None
        self.vectorizer = None
        self.model = None
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
        from .feature_engineering import TextPreprocessor
        self.preprocessor = TextPreprocessor()
    
    def _prepare_text_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        columns = ['item_id', 'param_1', 'param_2', 'param_3', 'title', 'description']
        
        train_subset = train_data[columns + [self.config.avito.TARGET_COLUMN]]
        test_subset = test_data[columns]
        
        def concatenate_params(row):
            return ' '.join([str(row['param_1']), str(row['param_2']), str(row['param_3'])])
        
        train_subset['title'] = train_subset['title'].fillna('none').apply(self.preprocessor.clean_text)
        train_subset['description'] = train_subset['description'].fillna('none').apply(self.preprocessor.clean_text)
        train_subset['params'] = train_subset.apply(concatenate_params, axis=1)
        
        test_subset['title'] = test_subset['title'].fillna('none').apply(self.preprocessor.clean_text)
        test_subset['description'] = test_subset['description'].fillna('none').apply(self.preprocessor.clean_text)
        test_subset['params'] = test_subset.apply(concatenate_params, axis=1)
        
        return train_subset, test_subset
    
    def _create_vectorizer(self, full_data: pd.DataFrame):
        features = ['title', 'description', 'params']
        text_data = full_data[features]
        
        def fetch(column):
            return lambda x: x[column]
        
        params = {
            'stop_words': self.preprocessor.stop_words,
            'analyzer': 'word',
            'token_pattern': r'\w{1,}',
            'sublinear_tf': self.config.model.TFIDF_SUBLINEAR_TF,
            'dtype': np.float32,
            'norm': self.config.model.TFIDF_NORM,
            'smooth_idf': self.config.model.TFIDF_SMOOTH_IDF,
            'ngram_range': self.config.model.TFIDF_NGRAM_RANGE
        }
        
        pipe = [
            ('description', TfidfVectorizer(
                **params, 
                preprocessor=fetch('description'), 
                max_features=self.config.model.TFIDF_MAX_FEATURES
            )),
            ('title', TfidfVectorizer(
                **params, 
                preprocessor=fetch('title')
            )),
            ('params', CountVectorizer(
                ngram_range=self.config.model.COUNT_VECTORIZER_NGRAM_RANGE,
                preprocessor=fetch('params')
            ))
        ]
        
        vectorizer = FeatureUnion(pipe)
        vectorizer.fit(text_data.to_dict('records'))
        return vectorizer
    
    def train_level1_model(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_data = self.data_loader.load_train_data()
        test_data = self.data_loader.load_test_data()
        
        train_idx = pd.read_csv(f'../../data/data/files/train_{fold}.csv')
        valid_idx = pd.read_csv(f'../../data/data/files/valid_{fold}.csv')
        
        train_subset = train_data.merge(train_idx, on=self.config.avito.ID_COLUMN)
        valid_subset = train_data.merge(valid_idx, on=self.config.avito.ID_COLUMN)
        
        train_processed, _ = self._prepare_text_data(train_subset, test_data)
        valid_processed, test_processed = self._prepare_text_data(valid_subset, test_data)
        
        full_data = pd.concat([train_processed, test_processed], ignore_index=True)
        vectorizer = self._create_vectorizer(full_data)
        
        train_matrix = vectorizer.transform(train_processed.to_dict('records'))
        valid_matrix = vectorizer.transform(valid_processed.to_dict('records'))
        test_matrix = vectorizer.transform(test_processed.to_dict('records'))
        
        model_params = {
            'alpha': self.config.model.RIDGE_ALPHA,
            'fit_intercept': True,
            'normalize': False,
            'copy_X': True,
            'max_iter': self.config.model.RIDGE_MAX_ITER,
            'tol': self.config.model.RIDGE_TOL,
            'solver': self.config.model.RIDGE_SOLVER,
            'random_state': self.config.avito.RANDOM_STATE
        }
        
        model = Ridge(**model_params)
        model.fit(train_matrix, train_subset[self.config.avito.TARGET_COLUMN])
        
        valid_driver = valid_subset[[self.config.avito.ID_COLUMN]].copy()
        test_driver = test_data[[self.config.avito.ID_COLUMN]].copy()
        
        valid_driver['ridge_score'] = model.predict(valid_matrix)
        test_driver['ridge_score'] = model.predict(test_matrix)
        
        return valid_driver, test_driver


class UserModel:
    def __init__(self, config):
        self.config = config
        self.data_loader = None
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
    
    def train_level1_model(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_file = f'{self.config.avito.FEATURES_DIR}/user/user_features.csv'
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"User features not found: {feature_file}")
        
        matrix = pd.read_csv(feature_file)
        actuals = self.data_loader.load_train_data([self.config.avito.ID_COLUMN, self.config.avito.TARGET_COLUMN])
        
        train_idx = pd.read_csv(f'../../data/data/files/train_{fold}.csv')
        valid_idx = pd.read_csv(f'../../data/data/files/valid_{fold}.csv')
        test_idx = self.data_loader.load_test_data([self.config.avito.ID_COLUMN])
        
        train_subset = matrix.merge(train_idx, on=self.config.avito.ID_COLUMN)
        valid_subset = matrix.merge(valid_idx, on=self.config.avito.ID_COLUMN)
        test_subset = matrix.merge(test_idx, on=self.config.avito.ID_COLUMN)
        
        labels = actuals.merge(train_idx, on=self.config.avito.ID_COLUMN)
        
        valid_driver = valid_subset[[self.config.avito.ID_COLUMN]].copy()
        test_driver = test_idx[[self.config.avito.ID_COLUMN]].copy()
        
        train_matrix = train_subset.iloc[:, 1:]
        valid_matrix = valid_subset.iloc[:, 1:]
        test_matrix = test_subset.iloc[:, 1:]
        
        model_params = {
            'alpha': 0.00000001,
            'fit_intercept': True,
            'normalize': False,
            'copy_X': True,
            'max_iter': None,
            'tol': 0.0001,
            'solver': 'auto',
            'random_state': self.config.avito.RANDOM_STATE
        }
        
        model = Ridge(**model_params)
        model.fit(train_matrix, labels[self.config.avito.TARGET_COLUMN])
        
        valid_driver['ridge_score'] = model.predict(valid_matrix)
        test_driver['ridge_score'] = model.predict(test_matrix)
        
        return valid_driver, test_driver


class EnsembleModel:
    def __init__(self, config):
        self.config = config
        self.data_loader = None
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
    
    def blend_predictions(self, model_predictions: Dict[str, pd.DataFrame], 
                         fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        actual = self.data_loader.load_train_data([self.config.avito.ID_COLUMN, self.config.avito.TARGET_COLUMN])
        actual = actual.rename(columns={self.config.avito.TARGET_COLUMN: 'actual'})
        
        train_idx = pd.read_csv(f'../../data/data/files/train_{fold}.csv')
        valid_idx = pd.read_csv(f'../../data/data/files/valid_{fold}.csv')
        test_idx = self.data_loader.load_test_data([self.config.avito.ID_COLUMN])
        
        train_data = actual.merge(train_idx, on=self.config.avito.ID_COLUMN)
        valid_data = actual.merge(valid_idx, on=self.config.avito.ID_COLUMN)
        
        features = []
        for model_name, predictions in model_predictions.items():
            train_data = train_data.merge(predictions, on=self.config.avito.ID_COLUMN, suffixes=('', f'_{model_name}'))
            valid_data = valid_data.merge(predictions, on=self.config.avito.ID_COLUMN, suffixes=('', f'_{model_name}'))
            test_idx = test_idx.merge(predictions, on=self.config.avito.ID_COLUMN, suffixes=('', f'_{model_name}'))
            features.append(f'ridge_score_{model_name}')
        
        model = LinearRegression(fit_intercept=True)
        model.fit(train_data[features], train_data['actual'])
        
        valid_data['score'] = model.predict(valid_data[features])
        test_idx['score'] = model.predict(test_idx[features])
        
        valid_data['score'] = valid_data['score'].clip(0, 1)
        test_idx['score'] = test_idx['score'].clip(0, 1)
        
        valid_result = valid_data[[self.config.avito.ID_COLUMN, 'actual', 'score']]
        test_result = test_idx[[self.config.avito.ID_COLUMN, 'score']]
        
        rmse = np.sqrt(mean_squared_error(valid_result['actual'], valid_result['score']))
        print(f'Fold {fold} RMSE: {rmse:.5f}')
        
        return valid_result, test_result


class ModelPipeline:
    def __init__(self, config):
        self.config = config
        self.data_loader = None
        
        self.text_model = TextModel(config)
        self.user_model = UserModel(config)
        self.ensemble_model = EnsembleModel(config)
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
        self.text_model.set_data_loader(data_loader)
        self.user_model.set_data_loader(data_loader)
        self.ensemble_model.set_data_loader(data_loader)
    
    def train_level1_models(self):
        print("Training Level 1 models...")
        
        all_valid_scores = []
        all_test_scores = []
        
        for fold in range(1, self.config.avito.N_FOLDS + 1):
            print(f"Training fold {fold}...")
            
            text_valid, text_test = self.text_model.train_level1_model(fold)
            user_valid, user_test = self.user_model.train_level1_model(fold)
            
            text_valid = text_valid.rename(columns={'ridge_score': 'text_score'})
            text_test = text_test.rename(columns={'ridge_score': 'text_score'})
            user_valid = user_valid.rename(columns={'ridge_score': 'user_score'})
            user_test = user_test.rename(columns={'ridge_score': 'user_score'})
            
            all_valid_scores.extend([text_valid, user_valid])
            all_test_scores.extend([text_test, user_test])
        
        self._save_level1_predictions(all_valid_scores, all_test_scores)
        print("Level 1 models training completed!")
    
    def _save_level1_predictions(self, valid_scores: List[pd.DataFrame], test_scores: List[pd.DataFrame]):
        os.makedirs('../../data/insample/scores', exist_ok=True)
        os.makedirs('../../data/outsample/scores', exist_ok=True)
        
        text_valid = pd.concat([df for df in valid_scores if 'text_score' in df.columns], ignore_index=True)
        text_test = pd.concat([df for df in test_scores if 'text_score' in df.columns], ignore_index=True)
        text_test = text_test.groupby(self.config.avito.ID_COLUMN)['text_score'].mean().reset_index()
        
        user_valid = pd.concat([df for df in valid_scores if 'user_score' in df.columns], ignore_index=True)
        user_test = pd.concat([df for df in test_scores if 'user_score' in df.columns], ignore_index=True)
        user_test = user_test.groupby(self.config.avito.ID_COLUMN)['user_score'].mean().reset_index()
        
        text_valid.to_csv('../../data/insample/scores/text_model.csv', index=False)
        text_test.to_csv('../../data/outsample/scores/text_model.csv', index=False)
        user_valid.to_csv('../../data/insample/scores/user_model.csv', index=False)
        user_test.to_csv('../../data/outsample/scores/user_model.csv', index=False)
    
    def train_ensemble_models(self):
        print("Training ensemble models...")
        
        all_valid_scores = []
        all_test_scores = []
        
        for fold in range(1, self.config.avito.N_FOLDS + 1):
            print(f"Training ensemble fold {fold}...")
            
            text_valid = pd.read_csv('../../data/insample/scores/text_model.csv')
            text_test = pd.read_csv('../../data/outsample/scores/text_model.csv')
            user_valid = pd.read_csv('../../data/insample/scores/user_model.csv')
            user_test = pd.read_csv('../../data/outsample/scores/user_model.csv')
            
            model_predictions = {
                'text': text_valid,
                'user': user_valid
            }
            
            valid_result, test_result = self.ensemble_model.blend_predictions(model_predictions, fold)
            all_valid_scores.append(valid_result)
            all_test_scores.append(test_result)
        
        self._save_ensemble_predictions(all_valid_scores, all_test_scores)
        print("Ensemble models training completed!")
    
    def _save_ensemble_predictions(self, valid_scores: List[pd.DataFrame], test_scores: List[pd.DataFrame]):
        train_data = pd.concat(valid_scores, ignore_index=True)
        test_data = pd.concat(test_scores, ignore_index=True)
        test_data = test_data.groupby(self.config.avito.ID_COLUMN)['score'].mean().reset_index()
        
        train_data = train_data[[self.config.avito.ID_COLUMN, 'score']].rename(columns={'score': self.config.avito.TARGET_COLUMN})
        test_data = test_data[[self.config.avito.ID_COLUMN, 'score']].rename(columns={'score': self.config.avito.TARGET_COLUMN})
        
        train_data.to_csv('../../data/insample/scores/ensemble_model.csv', index=False)
        test_data.to_csv('../../data/outsample/scores/ensemble_model.csv', index=False)
        
        rmse = np.sqrt(mean_squared_error(train_data[self.config.avito.TARGET_COLUMN], 
                                        train_data[self.config.avito.TARGET_COLUMN]))
        print(f'Final ensemble RMSE: {rmse:.5f}')
