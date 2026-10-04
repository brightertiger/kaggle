import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import xgboost as xgb

from .config import Config
from .data_utils import DataLoader, DataProcessor
from .feature_engineering import TextFeatureEngineer, NaiveBayesFeatureEngineer
from .models import XGBoostModel, NaiveBayesModel, NeuralNetworkModel

class SpookyAuthorPipeline:
    def __init__(self, data_dir: Path = None, model_dir: Path = None, score_dir: Path = None, config=None):
        self.config = config or Config()
        self.data_dir = Path(data_dir or self.config.DATA_DIR)
        self.model_dir = Path(model_dir or self.config.MODEL_DIR)
        self.score_dir = Path(score_dir or self.config.SCORE_DIR)
        
        self.data_loader = DataLoader(self.data_dir, self.config)
        self.text_feature_engineer = None
        self.nb_feature_engineer = NaiveBayesFeatureEngineer(self.config)
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.score_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Extracting text features...")
        
        train_data, test_data = self.data_loader.load_data()
        
        self.text_feature_engineer = TextFeatureEngineer(self.config)
        train_features = self.text_feature_engineer.extract_all_features(train_data)
        test_features = self.text_feature_engineer.extract_all_features(test_data)
        
        train_features.to_csv(self.score_dir / self.config.TRAIN_TEXT_FEATS, index=False)
        test_features.to_csv(self.score_dir / self.config.TEST_TEXT_FEATS, index=False)
        
        print(f"Text features saved: {train_features.shape[1]} features")
        return train_features, test_features
    
    def train_naive_bayes_models(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Training Naive Bayes models...")
        
        train_data, test_data = self.data_loader.load_data()
        train_data = self.data_loader.encode_authors(train_data)
        
        train_texts = train_data['text'].values.tolist()
        test_texts = test_data['text'].values.tolist()
        train_targets = train_data['author'].values
        
        nb_model = NaiveBayesModel(self.config)
        
        train_word_feats, test_word_feats = self.nb_feature_engineer.fit_transform_word_features(train_texts, test_texts)
        train_char_cnt_feats, test_char_cnt_feats = self.nb_feature_engineer.fit_transform_char_count_features(train_texts, test_texts)
        train_char_tf_feats, test_char_tf_feats = self.nb_feature_engineer.fit_transform_char_tfidf_features(train_texts, test_texts)
        
        train_word_score, test_word_score = nb_model.train_cv(train_word_feats, train_targets, test_word_feats)
        train_char_cnt_score, test_char_cnt_score = nb_model.train_cv(train_char_cnt_feats, train_targets, test_char_cnt_feats)
        train_char_tf_score, test_char_tf_score = nb_model.train_cv(train_char_tf_feats, train_targets, test_char_tf_feats)
        
        train_nb_score = pd.concat([
            train_data[['id', 'author']], train_word_score.add_prefix('word_'),
            train_char_cnt_score.add_prefix('char_count_'), train_char_tf_score.add_prefix('char_tfidf_')
        ], axis=1)
        test_nb_score = pd.concat([
            test_data[['id']], test_word_score.add_prefix('word_'),
            test_char_cnt_score.add_prefix('char_count_'), test_char_tf_score.add_prefix('char_tfidf_')
        ], axis=1)

        train_nb_score.to_csv(self.score_dir / self.config.TRAIN_NB_SCORE, index=False)
        test_nb_score.to_csv(self.score_dir / self.config.TEST_NB_SCORE, index=False)
        
        train_svd_feats, test_svd_feats = self.nb_feature_engineer.fit_transform_svd_features(train_texts, test_texts)
        train_svd_feats.insert(0, 'id', train_data['id'])
        test_svd_feats.insert(0, 'id', test_data['id'])
        train_svd_feats.to_csv(self.score_dir / self.config.TRAIN_NB_FEATS, index=False)
        test_svd_feats.to_csv(self.score_dir / self.config.TEST_NB_FEATS, index=False)
        
        print("Naive Bayes models trained and saved")
        return train_nb_score, test_nb_score
    
    def train_neural_network_models(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Training Neural Network models...")
        
        train_data, test_data = self.data_loader.load_data()
        train_data = self.data_loader.encode_authors(train_data)
        
        train_texts = train_data['text'].values.tolist()
        test_texts = test_data['text'].values.tolist()
        train_targets = train_data['author'].values.tolist()
        
        np.random.seed(self.config.RANDOM_STATE)
        
        simple_nn = NeuralNetworkModel(model_type='simple', config=self.config)
        train_nn_score, test_nn_score = simple_nn.train(train_texts, test_texts, train_targets)
        
        train_nn_final = pd.concat([train_data[['id']], train_nn_score], axis=1)
        test_nn_final = pd.concat([test_data[['id']], test_nn_score], axis=1)
        
        train_nn_final['keras'] = np.argmax(train_nn_score.values, axis=1)
        test_nn_final['keras'] = np.argmax(test_nn_score.values, axis=1)
        
        train_nn_final.to_csv(self.score_dir / self.config.TRAIN_NN_SCORE, index=False)
        test_nn_final.to_csv(self.score_dir / self.config.TEST_NN_SCORE, index=False)
        
        print("Simple Neural Network model trained and saved")
        return train_nn_final, test_nn_final
    
    def train_lstm_model(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Training LSTM model...")
        
        train_data, test_data = self.data_loader.load_data()
        train_data = self.data_loader.encode_authors(train_data)
        
        train_texts = train_data['text'].values.tolist()
        test_texts = test_data['text'].values.tolist()
        train_targets = train_data['author'].values.tolist()
        
        np.random.seed(self.config.RANDOM_STATE)
        
        lstm_model = NeuralNetworkModel(model_type='lstm', config=self.config)
        train_lstm_score, test_lstm_score = lstm_model.train(train_texts, test_texts, train_targets)
        
        train_lstm_final = pd.concat([train_data[['id']], train_lstm_score], axis=1)
        test_lstm_final = pd.concat([test_data[['id']], test_lstm_score], axis=1)
        
        train_lstm_final['lstm'] = np.argmax(train_lstm_score.values, axis=1)
        test_lstm_final['lstm'] = np.argmax(test_lstm_score.values, axis=1)
        
        train_lstm_final.to_csv(self.score_dir / self.config.TRAIN_LSTM_SCORE, index=False)
        test_lstm_final.to_csv(self.score_dir / self.config.TEST_LSTM_SCORE, index=False)
        
        print("LSTM model trained and saved")
        return train_lstm_final, test_lstm_final
    
    def train_xgboost_model(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Training XGBoost model...")
        
        data_processor = DataProcessor(self.data_dir, self.score_dir, self.config)
        train_data, test_data = data_processor.process_data()
        
        xgb_model = XGBoostModel(config=self.config)
        
        print("Running cross-validation...")
        self.cv_results = xgb_model.train_cv(train_data, target_column='author')
        self.cv_results.to_csv(self.score_dir / 'xgb_cv.csv', index=False)
        
        print("Training final model...")
        xgb_model.train(train_data, target_column='author')
        
        xgb_model.model.save_model(str(self.model_dir / self.config.XGB_MODEL))
        
        print("Making predictions...")
        predictions = xgb_model.predict(test_data.drop(columns=['id']))
        predictions = pd.concat([test_data[['id']], predictions], axis=1)
        
        predictions.to_csv(self.score_dir / self.config.XGB_SCORE, index=False)
        
        print("XGBoost model trained and saved")
        return train_data, predictions
    
    def run_full_pipeline(self) -> Tuple[List[float], pd.DataFrame]:
        print("Starting full Spooky Author Identification pipeline...")
        
        try:
            self.extract_text_features()
            self.train_naive_bayes_models()
            self.train_neural_network_models()
            self.train_lstm_model()
            train_data, predictions = self.train_xgboost_model()
            
            print("Pipeline completed successfully!")
            return self.cv_results['test-mlogloss-mean'].tolist(), predictions
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            raise
    
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        xgb_model = XGBoostModel(config=self.config)
        xgb_model.model = xgb.Booster()
        xgb_model.model.load_model(str(self.model_dir / self.config.XGB_MODEL))
        return xgb_model.get_feature_importance()
