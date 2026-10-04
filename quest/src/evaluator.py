import os
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from tqdm import tqdm

from .models import ModelFactory
from .data_utils import DataProcessor
from .config import Config
from .metrics import label_spearman, mean_spearman


class Evaluator:
    """Evaluation class for question understanding models"""

    def __init__(self, config, model_type: str = None):
        self.config = config
        self.device = config.resolve_device()
        self.model_type = model_type or config.model_type
        self.data_processor = DataProcessor(config)

    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        saved_config = Config(**checkpoint['config'])
        if saved_config.label_columns != self.config.label_columns:
            raise ValueError("Checkpoint target order differs from the current configuration")
        saved_config.pretrained = False
        saved_config.bert_config = checkpoint['bert_config']
        model = ModelFactory.create_model(checkpoint['model_type'], saved_config)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model

    def predict_fold(self, fold: int, data_dir: str, model_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for a specific fold"""
        # Load model
        model_path = os.path.join(model_dir, f"fold_{fold}", "best_model.pt")
        model = self.load_model(model_path)

        # Create data loader
        _, valid_loader = self.data_processor.create_data_loaders(
            fold, data_dir, self.config.batch_size
        )

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Predicting fold {fold}"):
                # Move to device
                question = batch['question'].to(self.device)
                answer = batch['answer'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                logits = model(question, answer)
                predictions = torch.sigmoid(logits)

                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Combine predictions
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)

        return predictions, labels

    def evaluate_fold(self, fold: int, data_dir: str, model_dir: str) -> float:
        """Evaluate a specific fold"""
        predictions, labels = self.predict_fold(fold, data_dir, model_dir)

        mean_correlation = mean_spearman(labels, predictions)
        print(f"Fold {fold} - Mean Spearman Correlation: {mean_correlation:.4f}")

        return mean_correlation

    def evaluate_all_folds(self, data_dir: str, model_dir: str) -> List[float]:
        """Evaluate all folds and return results"""
        fold_results = []

        for fold in range(1, self.config.n_folds + 1):
            correlation = self.evaluate_fold(fold, data_dir, model_dir)
            fold_results.append(correlation)

        mean_correlation = np.mean(fold_results)
        std_correlation = np.std(fold_results)

        print(f"Overall Results:")
        print(f"Mean Correlation: {mean_correlation:.4f} ± {std_correlation:.4f}")
        print(f"Individual Fold Results: {fold_results}")

        return fold_results

    def generate_oof_predictions(self, data_dir: str, model_dir: str) -> pd.DataFrame:
        """Join held-out rows by qa_id, restoring the original train.csv order."""
        frames = []
        for fold in range(1, self.config.n_folds + 1):
            predictions, _ = self.predict_fold(fold, data_dir, model_dir)
            ids = pd.read_csv(f"{data_dir}/split/text_valid_{fold}.csv")['qa_id']
            frame = pd.DataFrame(predictions, columns=self.config.label_columns)
            frame.insert(0, 'qa_id', ids.to_numpy())
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True).set_index('qa_id')
        train_ids = pd.read_csv(os.path.join(data_dir, 'train.csv'))['qa_id']
        if (not combined.index.is_unique or len(combined) != len(train_ids)
                or set(combined.index) != set(train_ids)):
            raise ValueError("Validation folds must cover each training qa_id exactly once")
        return combined.loc[train_ids].reset_index()

    def generate_ensemble_predictions(self, data_dir: str, model_dir: str) -> np.ndarray:
        """Compatibility alias: validation rows are OOF, never averaged across folds."""
        return self.generate_oof_predictions(data_dir, model_dir)[self.config.label_columns].to_numpy()

    def save_predictions(self, predictions: np.ndarray, labels: Optional[np.ndarray],
                        output_dir: str):
        """Save predictions and labels to files"""
        os.makedirs(output_dir, exist_ok=True)

        # Save as DataFrames
        pred_df = pd.DataFrame(predictions, columns=self.config.label_columns)

        pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
        if labels is not None:
            label_df = pd.DataFrame(labels, columns=self.config.label_columns)
            label_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

        print(f"Predictions saved to {output_dir}")

    def create_submission_file(self, test_predictions: np.ndarray,
                             test_ids: np.ndarray, output_path: str):
        """Create submission file for competition"""
        expected = (len(test_ids), self.config.num_labels)
        if test_predictions.shape != expected or not np.isfinite(test_predictions).all():
            raise ValueError(f"Expected finite predictions with shape {expected}")
        if ((test_predictions < 0) | (test_predictions > 1)).any():
            raise ValueError("Submission predictions must be in [0, 1]")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        submission_df = pd.DataFrame({
            'qa_id': test_ids,
            **{col: test_predictions[:, i] for i, col in enumerate(self.config.label_columns)}
        })

        submission_df.to_csv(output_path, index=False)
        print(f"Submission file saved to {output_path}")

    def analyze_predictions(self, predictions: np.ndarray, labels: np.ndarray):
        """Analyze prediction quality across different labels"""
        results = {}

        for idx, label_name in enumerate(self.config.label_columns):
            label = labels[:, idx]
            pred = predictions[:, idx]

            correlation = label_spearman(label, pred)
            mse = np.mean((label - pred) ** 2)

            results[label_name] = {
                'spearman_correlation': correlation,
                'mse': mse,
                'label_mean': np.mean(label),
                'pred_mean': np.mean(pred),
                'label_std': np.std(label),
                'pred_std': np.std(pred)
            }

        # Sort by correlation
        sorted_results = sorted(results.items(),
                              key=lambda x: x[1]['spearman_correlation'],
                              reverse=True)

        print("\nLabel-wise Performance Analysis:")
        print("-" * 80)
        print(f"{'Label':<40} {'Correlation':<12} {'MSE':<10}")
        print("-" * 80)

        for label_name, metrics in sorted_results:
            print(f"{label_name:<40} {metrics['spearman_correlation']:<12.4f} {metrics['mse']:<10.4f}")

        return results
