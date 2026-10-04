import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict

from .config import Config
from .models import ModelFactory, load_model_checkpoint
from .data_utils import create_data_loaders, create_test_loader


class ModelScorer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = ModelFactory.create_model(config, pretrained=False).to(self.device)
    
    def load_model(self, checkpoint_path: str) -> Dict:
        checkpoint_info = load_model_checkpoint(self.model, checkpoint_path)
        self.model.eval()
        return checkpoint_info
    
    def predict_batch(self, images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
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
        
        if not all_predictions:
            raise ValueError('Cannot predict an empty dataset')
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
        if not checkpoint_paths:
            raise ValueError('At least one checkpoint is required')
        test_loader = create_test_loader(self.config)
        all_predictions = []
        for checkpoint_path in checkpoint_paths:
            self.load_model(checkpoint_path)
            predictions = self.generate_predictions(test_loader)
            all_predictions.append(predictions.iloc[:, 1:].to_numpy())
        result_df = predictions.copy()
        result_df.iloc[:, 1:] = np.mean(all_predictions, axis=0)
        result_df.to_csv(output_path, index=False, compression='gzip')
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
        
        score_columns = [f'scr_{i}' for i in range(self.config.num_classes)]
        submission_df = predictions_df[['id']].copy()
        submission_df['attribute_ids'] = predictions_df[score_columns].apply(
            process_row, axis=1
        )
        
        return submission_df
    
    def create_submission(self, score_files: List[str], 
                         output_path: str) -> pd.DataFrame:
        return self.create_weighted_submission(score_files, [1.0] * len(score_files), output_path)

    def create_weighted_submission(self, score_files: List[str],
                                   weights: List[float], output_path: str) -> pd.DataFrame:
        if not score_files or len(score_files) != len(weights):
            raise ValueError('Provide one weight per score file, with at least one file')
        weights = np.asarray(weights, dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('Weights must be finite, nonnegative and have a positive sum')
        weights = weights / weights.sum()
        sample = pd.read_csv(self.config.sample_submission_path, dtype={'id': str})
        if sample.empty or sample['id'].duplicated().any():
            raise ValueError('Sample submission must contain unique, nonempty ids')
        ids = sample['id'].tolist()
        columns = [f'scr_{i}' for i in range(self.config.num_classes)]
        averaged = np.zeros((len(ids), len(columns)), dtype=np.float64)
        for score_file, weight in zip(score_files, weights):
            frame = pd.read_csv(score_file, dtype={'id': str})
            if frame['id'].duplicated().any() or set(frame['id']) != set(ids):
                raise ValueError(f'{score_file} must contain exactly the test image ids')
            values = frame.set_index('id').loc[ids, columns].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f'{score_file} contains invalid probabilities')
            averaged += values * weight
        ensemble = pd.DataFrame(averaged, columns=columns)
        ensemble.insert(0, 'id', ids)
        submission = self.process_predictions(ensemble)
        submission.to_csv(output_path, index=False)
        print(f'Saved {len(submission)} test predictions to {output_path}')
        return submission


class EnsembleScorer:
    def __init__(self, config: Config):
        self.config = config
        self._scorer = None
        self.submission_generator = SubmissionGenerator(config)
    
    @property
    def scorer(self):
        if self._scorer is None:
            self._scorer = ModelScorer(self.config)
        return self._scorer

    def score_all_folds(self) -> List[str]:
        score_files = []
        
        folds_to_score = range(1, self.config.num_folds + 1) if self.config.fold_idx is None else [self.config.fold_idx]
        
        for fold_idx in folds_to_score:
            checkpoint_path = self.config.get_model_path(fold_idx, 'stage_2')
            score_path = self.config.get_score_path(fold_idx)
            
            if os.path.exists(checkpoint_path):
                self.scorer.score_test_set([checkpoint_path], score_path)
                score_files.append(score_path)
            else:
                raise FileNotFoundError(f"Checkpoint not found for fold {fold_idx}: {checkpoint_path}")
        
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
