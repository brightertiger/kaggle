import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any, List
import os
from .config import Config, get_config
from .data_utils import create_data_loaders, create_submission_dataset, read_data
from .trainer import TweetTrainer
from .evaluator import TweetScorer, TweetEvaluator

class TweetSentimentPipeline:
    
    def __init__(self, config: Config = None):
        self.config = config or get_config()
        self.trainer = TweetTrainer(self.config)
        self.scorer = TweetScorer(self.config)
        self.evaluator = TweetEvaluator(self.config)
    
    def create_folds(self, data_path: str) -> str:
        data = read_data(data_path, labeled=True)
        if data.empty or data['sentiment'].value_counts().min() < self.config.data.n_folds:
            raise ValueError('Each sentiment needs at least n_folds labeled rows')
        data['subset'] = -1
        
        skf = StratifiedKFold(
            n_splits=self.config.data.n_folds,
            random_state=self.config.data.random_seed,
            shuffle=True
        )
        
        fold = 0
        for train_idx, valid_idx in skf.split(data.textID, data.sentiment):
            data.loc[valid_idx, 'subset'] = fold
            fold += 1
        
        processed_path = os.path.join(self.config.data.processed_path, 'train.csv')
        os.makedirs(self.config.data.processed_path, exist_ok=True)
        data.to_csv(processed_path, index=False)
        self.config.data.train_path = processed_path
        
        print(f"Created {self.config.data.n_folds} folds. Data saved to {processed_path}")
        return processed_path
    
    def train_all_folds(self, data_path: str) -> Dict[str, float]:
        self.config.data.train_path = data_path
        fold_scores = {}
        
        for fold in range(self.config.data.n_folds):
            print(f"\nTraining fold {fold + 1}/{self.config.data.n_folds}")
            
            train_loader, valid_loader = create_data_loaders(self.config, fold)
            best_loss = self.trainer.train_fold(fold, train_loader, valid_loader)
            
            print(f"Fold {fold} completed. Best validation loss: {best_loss:.4f}")
            fold_scores[f'fold_{fold}'] = best_loss
        
        return fold_scores
    
    def evaluate_all_folds(self, data_path: str) -> Dict[str, float]:
        self.config.data.train_path = data_path
        fold_scores = {}
        
        for fold in range(self.config.data.n_folds):
            print(f"\nEvaluating fold {fold}")
            
            _, valid_loader = create_data_loaders(self.config, fold)
            predictions = self.scorer.predict_fold(fold, valid_loader)
            
            valid_data = valid_loader.dataset.data
            
            score = self.evaluator.evaluate_fold(fold, valid_data, predictions)
            fold_scores[f'fold_{fold}'] = score
            
            print(f"Fold {fold} Jaccard score: {score:.4f}")
        
        return fold_scores
    
    def run_full_pipeline(self, data_path: str) -> Dict[str, Any]:
        print("Starting Tweet Sentiment Extraction Pipeline")
        print("=" * 50)
        
        processed_data_path = self.create_folds(data_path)
        
        print("\nTraining models...")
        training_results = self.train_all_folds(processed_data_path)
        
        print("\nEvaluating models...")
        evaluation_results = self.evaluate_all_folds(processed_data_path)
        
        avg_score = np.mean(list(evaluation_results.values()))
        print(f"\nAverage Jaccard Score: {avg_score:.4f}")
        
        results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'average_score': avg_score,
            'processed_data_path': processed_data_path
        }
        
        return results
    
    def predict_test_set(self, test_path: str) -> pd.DataFrame:
        print("Generating predictions for test set...")
        
        test_loader = create_submission_dataset(self.config, test_path)
        all_predictions = []
        
        for fold in range(self.config.data.n_folds):
            print(f"Predicting with fold {fold}")
            fold_predictions = self.scorer.predict_fold(fold, test_loader)
            all_predictions.append(fold_predictions)
        
        ensemble_predictions = self._ensemble_predictions(all_predictions)
        
        test_data = test_loader.dataset.data
        selected_text = []
        for row, prediction in zip(test_data.itertuples(), ensemble_predictions.itertuples()):
            offsets = self.evaluator.get_offsets(row.text, row.sentiment)
            selected_text.append(self.evaluator.extract_selected_text(
                row.text, prediction.start_pred, prediction.end_pred, offsets))
        submission = pd.DataFrame({
            'textID': test_data['textID'],
            'selected_text': selected_text
        })
        
        return submission
    
    def _ensemble_predictions(self, predictions_list: List[pd.DataFrame]) -> pd.DataFrame:
        start_preds = np.mean([pred['start_pred'].values for pred in predictions_list], axis=0)
        end_preds = np.mean([pred['end_pred'].values for pred in predictions_list], axis=0)
        
        start_preds = np.round(start_preds).astype(int)
        end_preds = np.round(end_preds).astype(int)
        
        ensemble_df = pd.DataFrame({
            'start_pred': start_preds,
            'end_pred': end_preds
        })
        
        return ensemble_df
