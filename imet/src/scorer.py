import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Optional

from .config import Config
from .models import ResNextClassifier, ModelFactory, load_model_checkpoint
from .data_utils import create_test_loader


class ModelScorer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = ModelFactory.create_model(config).to(self.device)
    
    def load_model(self, checkpoint_path: str) -> Dict:
        checkpoint_info = load_model_checkpoint(self.model, checkpoint_path)
        self.model.eval()
        return checkpoint_info
    
    def predict_batch(self, images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.sigmoid(outputs)
            return probabilities.cpu().numpy()
    
    def generate_predictions(self, data_loader: DataLoader) -> pd.DataFrame:
        all_predictions = []
        all_ids = []
        
        self.model.eval()
        
        for batch in tqdm(data_loader, desc="Generating predictions"):
            images = batch['image'].to(self.device)
            batch_ids = batch['idx']
            
            predictions = self.predict_batch(images)
            
            all_predictions.append(predictions)
            all_ids.extend(batch_ids)
        
        all_predictions = np.vstack(all_predictions)
        
        prediction_df = pd.DataFrame(all_predictions)
        prediction_df.columns = [f'scr_{i}' for i in range(all_predictions.shape[1])]
        
        id_df = pd.DataFrame({'id': all_ids})
        result_df = pd.concat([id_df, prediction_df], axis=1)
        
        return result_df
    
    def score_fold(self, fold_idx: int, checkpoint_path: str, 
                   output_path: str) -> pd.DataFrame:
        print(f"Scoring fold {fold_idx}...")
        
        checkpoint_info = self.load_model(checkpoint_path)
        print(f"Loaded model: Loss={checkpoint_info['loss']:.4f}, "
              f"Metric={checkpoint_info['metric']:.4f}")
        
        train_loader, valid_loader = create_data_loaders(self.config, fold_idx)
        predictions_df = self.generate_predictions(valid_loader)
        
        predictions_df.to_csv(output_path, index=False, compression='gzip')
        print(f"Saved predictions to {output_path}")
        
        return predictions_df
    
    def score_test_set(self, checkpoint_paths: List[str], 
                      output_path: str) -> pd.DataFrame:
        print("Scoring test set...")
        
        test_loader = create_test_loader(self.config)
        all_predictions = []
        
        for i, checkpoint_path in enumerate(checkpoint_paths):
            print(f"Loading model {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")
            
            checkpoint_info = self.load_model(checkpoint_path)
            print(f"Model {i+1}: Loss={checkpoint_info['loss']:.4f}, "
                  f"Metric={checkpoint_info['metric']:.4f}")
            
            predictions = []
            for batch in tqdm(test_loader, desc=f"Model {i+1} predictions"):
                images = batch['image'].to(self.device)
                batch_ids = batch['idx']
                
                batch_predictions = self.predict_batch(images)
                predictions.append(batch_predictions)
            
            predictions = np.vstack(predictions)
            all_predictions.append(predictions)
        
        ensemble_predictions = np.mean(all_predictions, axis=0)
        
        prediction_df = pd.DataFrame(ensemble_predictions)
        prediction_df.columns = [f'scr_{i}' for i in range(ensemble_predictions.shape[1])]
        
        test_ids = []
        for batch in test_loader:
            test_ids.extend(batch['idx'])
        
        id_df = pd.DataFrame({'id': test_ids})
        result_df = pd.concat([id_df, prediction_df], axis=1)
        
        result_df.to_csv(output_path, index=False, compression='gzip')
        print(f"Saved ensemble predictions to {output_path}")
        
        return result_df


class SubmissionGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.min_threshold = config.min_threshold
        self.top_k = config.top_k
    
    def process_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        def process_row(row):
            row_values = row.values
            candidate_indices = np.argsort(row_values)[-self.top_k:]
            
            labels = []
            for idx in candidate_indices:
                if row_values[idx] > self.min_threshold:
                    labels.append(str(idx))
            
            if len(labels) == 0:
                labels.append(str(np.argmax(row_values)))
            
            return ' '.join(labels)
        
        submission_df = predictions_df[['id']].copy()
        submission_df['attribute_ids'] = predictions_df.iloc[:, 1:].apply(
            process_row, axis=1
        )
        
        return submission_df
    
    def create_submission(self, score_files: List[str], 
                         output_path: str) -> pd.DataFrame:
        print("Creating submission file...")
        
        all_predictions = []
        
        for score_file in score_files:
            print(f"Loading {score_file}...")
            df = pd.read_csv(score_file, compression='gzip')
            df.iloc[:, 1:] = df.iloc[:, 1:].astype(np.float16)
            all_predictions.append(df)
        
        combined_df = pd.concat(all_predictions, ignore_index=True)
        ensemble_df = combined_df.groupby('id').mean().reset_index()
        
        submission_df = self.process_predictions(ensemble_df)
        
        sample_submission = pd.read_csv(self.config.sample_submission_path)
        final_submission = submission_df.append(sample_submission)
        final_submission = final_submission.drop_duplicates(subset=['id'], keep='first')
        
        final_submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        print(f"Total predictions: {len(final_submission)}")
        
        return final_submission
    
    def create_weighted_submission(self, score_files: List[str], 
                                 weights: List[float], 
                                 output_path: str) -> pd.DataFrame:
        print("Creating weighted submission file...")
        
        if len(score_files) != len(weights):
            raise ValueError("Number of score files must match number of weights")
        
        all_predictions = []
        
        for score_file, weight in zip(score_files, weights):
            print(f"Loading {score_file} with weight {weight}...")
            df = pd.read_csv(score_file, compression='gzip')
            df.iloc[:, 1:] = df.iloc[:, 1:] * weight
            all_predictions.append(df)
        
        combined_df = pd.concat(all_predictions, ignore_index=True)
        ensemble_df = combined_df.groupby('id').sum().reset_index()
        
        submission_df = self.process_predictions(ensemble_df)
        
        sample_submission = pd.read_csv(self.config.sample_submission_path)
        final_submission = submission_df.append(sample_submission)
        final_submission = final_submission.drop_duplicates(subset=['id'], keep='first')
        
        final_submission.to_csv(output_path, index=False)
        print(f"Weighted submission saved to {output_path}")
        print(f"Total predictions: {len(final_submission)}")
        
        return final_submission


class EnsembleScorer:
    def __init__(self, config: Config):
        self.config = config
        self.scorer = ModelScorer(config)
        self.submission_generator = SubmissionGenerator(config)
    
    def score_all_folds(self) -> List[str]:
        score_files = []
        
        folds_to_score = range(1, self.config.num_folds + 1) if self.config.fold_idx is None else [self.config.fold_idx]
        
        for fold_idx in folds_to_score:
            checkpoint_path = self.config.get_model_path(fold_idx, 'stage_2')
            score_path = self.config.get_score_path(fold_idx)
            
            if os.path.exists(checkpoint_path):
                self.scorer.score_fold(fold_idx, checkpoint_path, score_path)
                score_files.append(score_path)
            else:
                print(f"Warning: Checkpoint not found for fold {fold_idx}: {checkpoint_path}")
        
        return score_files
    
    def create_final_submission(self, score_files: List[str], 
                              output_path: str) -> pd.DataFrame:
        return self.submission_generator.create_submission(score_files, output_path)
    
    def create_weighted_submission(self, score_files: List[str], 
                                 weights: List[float], 
                                 output_path: str) -> pd.DataFrame:
        return self.submission_generator.create_weighted_submission(
            score_files, weights, output_path
        )
