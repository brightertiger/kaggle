import pandas as pd
from sklearn.metrics import log_loss

from .config import Config
from .data_utils import DataProcessor
from .models import CNNBasic, CNNAdvanced, VGG16Model, EnsembleModel
from .trainer import ModelTrainer
from .predictor import ModelPredictor
from .feature_engineering import FeatureEngineer

class IcebergPipeline:
    def __init__(self, config=None):
        self.config = config if config is not None else Config()
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
        
        train_ids = pd.read_json(f'{self.config.DATA_DIR}/download/train.json')['id']
        test_ids = pd.read_json(f'{self.config.DATA_DIR}/download/test.json')['id']
        train_scores, test_scores = {}, {}
        train_labels = None
        for model_file in model_files:
            train_model = self._aligned_csv(f'{self.config.DATA_DIR}/model/{model_file}', train_ids)
            test_model = self._aligned_csv(f'{self.config.SUBMISSION_DIR}/{model_file}', test_ids)
            if train_labels is not None and not train_model['label'].equals(train_labels):
                raise ValueError('Model OOF labels do not match')
            train_labels = train_model['label']
            train_scores[model_file] = train_model['score']
            test_scores[model_file] = test_model['is_iceberg']
        train_stacked, test_stacked = self.ensemble.simple_stack(
            pd.DataFrame(train_scores), pd.DataFrame(test_scores))
        print(f'Ensemble OOF Log Loss: {log_loss(train_labels, train_stacked):.6f}')
        ensemble_submission = pd.DataFrame({'id': test_ids.to_numpy(),
                                            'is_iceberg': test_stacked.to_numpy()})
        ensemble_submission.to_csv(f'{self.config.SUBMISSION_DIR}/ensemble.csv', index=False)

        print("Ensemble creation completed!")
    
    def create_xgboost_features(self):
        print("Creating XGBoost features...")
        
        train_file = f'{self.config.DATA_DIR}/download/train.json'
        test_file = f'{self.config.DATA_DIR}/download/test.json'
        
        self.feature_engineer.create_xgboost_features(train_file, test_file)
        
        print("XGBoost feature creation completed!")
    
    @staticmethod
    def _aligned_csv(path, ids):
        frame = pd.read_csv(path, dtype={'id': str}).set_index('id', verify_integrity=True)
        ids = ids.astype(str)
        if set(frame.index) != set(ids):
            raise ValueError(f'Prediction IDs do not match input IDs: {path}')
        return frame.loc[ids].reset_index()

    def create_xgboost_submission(self):
        train = pd.read_csv(f'{self.config.DATA_DIR}/train_xgb.csv', dtype={'id': str})
        test = pd.read_csv(f'{self.config.DATA_DIR}/test_xgb.csv', dtype={'id': str})
        labels = pd.read_json(f'{self.config.DATA_DIR}/download/train.json')[['id', 'is_iceberg']]
        labels['id'] = labels['id'].astype(str)
        train = train.merge(labels.rename(columns={'is_iceberg': 'label'}), on='id',
                            validate='one_to_one')
        for name in ('cnn_basic', 'cnn_advanced', 'vgg16'):
            oof = self._aligned_csv(f'{self.config.DATA_DIR}/model/{name}.csv', train['id'])
            scores = self._aligned_csv(f'{self.config.SUBMISSION_DIR}/{name}.csv', test['id'])
            train[name] = oof['score'].to_numpy()
            test[name] = scores['is_iceberg'].to_numpy()
        predictions = self.ensemble.xgboost_stack(train, test)
        submission = pd.DataFrame({'id': test['id'], 'is_iceberg': predictions})
        submission.to_csv(f'{self.config.SUBMISSION_DIR}/xgboost.csv', index=False)
        print('XGBoost submission written.')
        return submission

    def run_full_pipeline(self):
        print('Starting Iceberg Classification Pipeline...', flush=True)
        self.prepare_data()
        self.create_xgboost_features()
        self.train_models()
        self.generate_predictions()
        self.create_ensemble()
        self.create_xgboost_submission()
        print('Pipeline completed successfully!')
