import os
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

from .config import Config
from .data_utils import DataProcessor
from .models import CNNBasic, CNNAdvanced, VGG16Model, EnsembleModel
from .trainer import ModelTrainer
from .predictor import ModelPredictor
from .feature_engineering import FeatureEngineer

class IcebergPipeline:
    def __init__(self):
        self.config = Config()
        self.data_processor = DataProcessor(self.config)
        self.trainer = ModelTrainer(self.config)
        self.predictor = ModelPredictor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.ensemble = EnsembleModel(self.config)
        
    def prepare_data(self):
        print("Preparing data...")
        
        self.data_processor.process_train_data(
            'source_1', 
            self.data_processor.convert_images_source1
        )
        self.data_processor.process_test_data(
            'source_1', 
            self.data_processor.convert_images_source1
        )
        
        self.data_processor.process_train_data(
            'source_2', 
            self.data_processor.convert_images_source2
        )
        self.data_processor.process_test_data(
            'source_2', 
            self.data_processor.convert_images_source2
        )
        
        print("Data preparation completed!")
    
    def train_models(self):
        print("Training models...")
        
        cnn_basic = CNNBasic(self.config)
        cnn_advanced = CNNAdvanced(self.config)
        vgg16_model = VGG16Model(self.config)
        
        self.trainer.train_model(
            cnn_basic, 
            'cnn_basic', 
            'source_1', 
            self.config.IMAGE_TRANSFORMS['source_1']
        )
        
        self.trainer.train_model(
            cnn_advanced, 
            'cnn_advanced', 
            'source_1', 
            self.config.IMAGE_TRANSFORMS['source_2']
        )
        
        self.trainer.train_vgg16_model(
            vgg16_model, 
            'vgg16', 
            'source_1', 
            self.config.IMAGE_TRANSFORMS['source_1']
        )
        
        print("Model training completed!")
    
    def generate_predictions(self):
        print("Generating predictions...")
        
        cnn_basic = CNNBasic(self.config)
        cnn_advanced = CNNAdvanced(self.config)
        vgg16_model = VGG16Model(self.config)
        
        self.predictor.predict_test_set(cnn_basic, 'cnn_basic', 'source_1')
        self.predictor.predict_cv_set(cnn_basic, 'cnn_basic', 'source_1')
        
        self.predictor.predict_test_set(cnn_advanced, 'cnn_advanced', 'source_1')
        self.predictor.predict_cv_set(cnn_advanced, 'cnn_advanced', 'source_1')
        
        self.predictor.predict_test_set(vgg16_model, 'vgg16', 'source_1')
        self.predictor.predict_cv_set(vgg16_model, 'vgg16', 'source_1')
        
        print("Prediction generation completed!")
    
    def create_ensemble(self):
        print("Creating ensemble...")
        
        model_files = [
            'cnn_basic.csv',
            'cnn_advanced.csv', 
            'vgg16.csv'
        ]
        
        train_scores = []
        test_scores = []
        
        for model_file in model_files:
            train_model = pd.read_csv(f'{self.config.DATA_DIR}/model/{model_file}')
            test_model = pd.read_csv(f'{self.config.SUBMISSION_DIR}/{model_file}')
            
            train_scores.append(train_model['score'])
            test_scores.append(test_model['is_iceberg'])
        
        train_scores = pd.DataFrame(train_scores).T
        test_scores = pd.DataFrame(test_scores).T
        
        train_stacked, test_stacked = self.ensemble.simple_stack(train_scores, test_scores)
        
        train_labels = pd.read_csv(f'{self.config.DATA_DIR}/model/cnn_basic.csv')['label']
        ensemble_loss = log_loss(train_labels, train_stacked)
        print(f"Ensemble Log Loss: {ensemble_loss:.6f}")
        
        ensemble_submission = pd.DataFrame({
            'id': pd.read_csv(f'{self.config.SUBMISSION_DIR}/cnn_basic.csv')['id'],
            'is_iceberg': test_stacked
        })
        
        ensemble_submission.to_csv(f'{self.config.SUBMISSION_DIR}/ensemble.csv', index=False)
        
        print("Ensemble creation completed!")
    
    def create_xgboost_features(self):
        print("Creating XGBoost features...")
        
        train_file = f'{self.config.DATA_DIR}/download/train.json'
        test_file = f'{self.config.DATA_DIR}/download/test.json'
        
        self.feature_engineer.create_xgboost_features(train_file, test_file)
        
        print("XGBoost feature creation completed!")
    
    def run_full_pipeline(self):
        print("Starting Iceberg Classification Pipeline...")
        
        self.prepare_data()
        self.train_models()
        self.generate_predictions()
        self.create_ensemble()
        self.create_xgboost_features()
        
        print("Pipeline completed successfully!")
