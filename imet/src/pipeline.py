import os
import time
import logging
from typing import Dict, List, Optional

from .config import Config
from .data_utils import DataPreprocessor
from .trainer import ModelTrainer
from .scorer import EnsembleScorer


class IMetPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        
        self.preprocessor = DataPreprocessor(config)
        self.trainer = ModelTrainer(config)
        self.scorer = EnsembleScorer(config)
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.logs_dir, 'pipeline.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def preprocess_data(self) -> None:
        self.logger.info("🔄 Starting data preprocessing...")
        start_time = time.time()
        
        try:
            self.preprocessor.create_folds()
            preprocessing_time = time.time() - start_time
            self.logger.info(f"✅ Data preprocessing completed in {preprocessing_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"❌ Data preprocessing failed: {str(e)}")
            raise
    
    def train_model(self) -> Dict[int, Dict[str, float]]:
        self.logger.info("🚀 Starting model training...")
        start_time = time.time()
        
        try:
            if not os.path.exists(self.config.folds_csv_path):
                self.logger.warning("Folds file not found. Running preprocessing first...")
                self.preprocess_data()
            
            results = self.trainer.train_all_folds()
            
            training_time = time.time() - start_time
            self.logger.info(f"✅ Model training completed in {training_time:.2f} seconds")
            
            self._log_training_summary(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Model training failed: {str(e)}")
            raise
    
    def generate_predictions(self) -> None:
        self.logger.info("🔮 Starting prediction generation...")
        start_time = time.time()
        
        try:
            score_files = self.scorer.score_all_folds()
            
            if not score_files:
                self.logger.error("No score files generated. Check if models are trained.")
                return
            
            submission_path = self.config.get_submission_path('submission')
            self.scorer.create_final_submission(score_files, submission_path)
            
            prediction_time = time.time() - start_time
            self.logger.info(f"✅ Prediction generation completed in {prediction_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"❌ Prediction generation failed: {str(e)}")
            raise
    
    def generate_weighted_predictions(self, weights: List[float]) -> None:
        self.logger.info("🔮 Starting weighted prediction generation...")
        start_time = time.time()
        
        try:
            score_files = self.scorer.score_all_folds()
            
            if not score_files:
                self.logger.error("No score files generated. Check if models are trained.")
                return
            
            if len(score_files) != len(weights):
                self.logger.error(f"Number of score files ({len(score_files)}) "
                                f"must match number of weights ({len(weights)})")
                return
            
            submission_path = self.config.get_submission_path('weighted_submission')
            self.scorer.create_weighted_submission(score_files, weights, submission_path)
            
            prediction_time = time.time() - start_time
            self.logger.info(f"✅ Weighted prediction generation completed in {prediction_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"❌ Weighted prediction generation failed: {str(e)}")
            raise
    
    def run_complete_pipeline(self) -> Dict[int, Dict[str, float]]:
        self.logger.info("🎯 Running complete pipeline...")
        pipeline_start_time = time.time()
        
        try:
            self.preprocess_data()
            training_results = self.train_model()
            self.generate_predictions()
            
            pipeline_time = time.time() - pipeline_start_time
            self.logger.info(f"🎉 Complete pipeline finished in {pipeline_time:.2f} seconds")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ Complete pipeline failed: {str(e)}")
            raise
    
    def _log_training_summary(self, results: Dict[int, Dict[str, float]]) -> None:
        self.logger.info("📊 Training Summary:")
        
        if not results:
            self.logger.warning("No training results to summarize")
            return
        
        best_fbeta_scores = [result['best_fbeta'] for result in results.values()]
        avg_fbeta = sum(best_fbeta_scores) / len(best_fbeta_scores)
        max_fbeta = max(best_fbeta_scores)
        min_fbeta = min(best_fbeta_scores)
        
        self.logger.info(f"  Average F-Beta Score: {avg_fbeta:.4f}")
        self.logger.info(f"  Best F-Beta Score: {max_fbeta:.4f}")
        self.logger.info(f"  Worst F-Beta Score: {min_fbeta:.4f}")
        
        for fold_idx, result in results.items():
            self.logger.info(f"  Fold {fold_idx}: F-Beta = {result['best_fbeta']:.4f}")
    
    def validate_setup(self) -> bool:
        self.logger.info("🔍 Validating setup...")
        
        required_files = [
            self.config.train_csv_path,
            self.config.subset_csv_path,
            self.config.sample_submission_path
        ]
        
        required_dirs = [
            self.config.train_images_path,
            self.config.test_images_path
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
        
        if missing_files:
            self.logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        if missing_dirs:
            self.logger.error(f"❌ Missing required directories: {missing_dirs}")
            return False
        
        self.logger.info("✅ Setup validation passed")
        return True
    
    def get_model_info(self) -> Dict[str, any]:
        model_info = {
            'model_name': self.config.model_name,
            'num_classes': self.config.num_classes,
            'image_size': self.config.image_size,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'learning_rate': self.config.learning_rate,
            'device': self.config.device
        }
        
        trained_models = []
        for fold_idx in range(1, self.config.num_folds + 1):
            model_path = self.config.get_model_path(fold_idx, 'stage_2')
            if os.path.exists(model_path):
                trained_models.append(fold_idx)
        
        model_info['trained_folds'] = trained_models
        model_info['total_folds'] = self.config.num_folds
        
        return model_info
