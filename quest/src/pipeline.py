import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from .config import Config
from .data_utils import DataProcessor
from .trainer import Trainer
from .evaluator import Evaluator
from .models import ModelFactory


class QuestionUnderstandingPipeline:
    """Main pipeline class for question understanding model training and inference"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor(config)
        
    def prepare_data(self, data_path: str):
        """Prepare and split data for training"""
        print("Preparing data...")
        
        # Load training data
        train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
        train_data = train_data.sample(frac=1.0, random_state=2017).reset_index(drop=True)
        
        # Create folds
        folds = self.data_processor.create_folds(train_data, self.config.n_folds)
        
        # Save split data
        for fold_idx, (train_idx, valid_idx) in enumerate(folds, 1):
            train_fold, valid_fold = self.data_processor.split_data(
                train_data, train_idx, valid_idx
            )
            self.data_processor.save_split_data(
                train_fold, valid_fold, fold_idx, data_path
            )
            print(f"Fold {fold_idx}: Train {len(train_fold)}, Valid {len(valid_fold)}")
        
        # Prepare test data
        test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
        
        # Save test data splits
        test_text = test_data[['qa_id'] + self.config.text_columns]
        test_meta = test_data[['qa_id'] + self.config.user_columns + 
                             self.config.url_columns + self.config.categorical_columns]
        
        os.makedirs(f"{data_path}/split", exist_ok=True)
        test_text.to_csv(f"{data_path}/split/text_score.csv", index=False)
        test_meta.to_csv(f"{data_path}/split/data_score.csv", index=False)
        
        print("Data preparation completed!")
    
    def train_fold(self, fold: int, data_path: str, model_path: str) -> float:
        """Train model for a specific fold"""
        trainer = Trainer(self.config)
        val_loss = trainer.train_fold(fold, data_path, model_path)
        return val_loss
    
    def train_all_folds(self, data_path: str, model_path: str) -> List[float]:
        """Train models for all folds"""
        print("Starting training for all folds...")
        
        # Prepare data if not already done
        if not os.path.exists(f"{data_path}/split"):
            self.prepare_data(data_path)
        
        fold_results = []
        
        for fold in range(1, self.config.n_folds + 1):
            print(f"\n{'='*50}")
            print(f"Training Fold {fold}/{self.config.n_folds}")
            print(f"{'='*50}")
            
            val_loss = self.train_fold(fold, data_path, model_path)
            fold_results.append(val_loss)
            
            print(f"Fold {fold} completed with validation loss: {val_loss:.4f}")
        
        print(f"\n{'='*50}")
        print("Training Summary")
        print(f"{'='*50}")
        print(f"Average validation loss: {np.mean(fold_results):.4f}")
        print(f"Standard deviation: {np.std(fold_results):.4f}")
        
        return fold_results
    
    def inference(self, data_path: str, model_path: str, output_path: Optional[str] = None):
        """Generate predictions on test data"""
        print("Starting inference...")
        
        if output_path is None:
            output_path = self.config.output_dir
        
        evaluator = Evaluator(self.config)
        
        # Generate predictions for each fold
        all_predictions = []
        
        for fold in range(1, self.config.n_folds + 1):
            print(f"Generating predictions for fold {fold}...")
            
            # Load model
            model_path_fold = os.path.join(model_path, f"fold_{fold}", "best_model.pt")
            model = evaluator.load_model(model_path_fold)
            
            # Load test data
            test_text = pd.read_csv(f"{data_path}/split/text_score.csv")
            test_meta = pd.read_csv(f"{data_path}/split/data_score.csv")
            
            # Create dummy labels for test data
            dummy_labels = pd.DataFrame(
                np.zeros((len(test_text), len(self.config.label_columns))),
                columns=self.config.label_columns
            )
            dummy_labels['qa_id'] = test_text['qa_id']
            
            # Create dataset
            from .data_utils import QuestionAnswerDataset
            test_dataset = QuestionAnswerDataset(
                test_text, test_meta, dummy_labels,
                evaluator.data_processor.tokenizer, self.config.max_length
            )
            
            # Generate predictions
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.config.batch_size, shuffle=False
            )
            
            predictions = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Fold {fold}"):
                    question = batch['question'].to(evaluator.device)
                    answer = batch['answer'].to(evaluator.device)
                    
                    logits = model(question, answer)
                    pred = torch.sigmoid(logits)
                    predictions.append(pred.cpu().numpy())
            
            fold_predictions = np.vstack(predictions)
            all_predictions.append(fold_predictions)
        
        # Ensemble predictions
        ensemble_predictions = np.mean(all_predictions, axis=0)
        
        # Create submission file
        test_ids = test_text['qa_id'].values
        submission_path = os.path.join(output_path, "submission.csv")
        evaluator.create_submission_file(
            ensemble_predictions, test_ids, submission_path
        )
        
        print(f"Inference completed! Submission saved to {submission_path}")
        return ensemble_predictions
    
    def evaluate_cv_performance(self, data_path: str, model_path: str):
        """Evaluate cross-validation performance"""
        print("Evaluating cross-validation performance...")
        
        evaluator = Evaluator(self.config)
        fold_results = evaluator.evaluate_all_folds(data_path, model_path)
        
        # Generate and save ensemble predictions for validation
        ensemble_predictions = evaluator.generate_ensemble_predictions(data_path, model_path)
        
        # Save predictions
        output_dir = os.path.join(model_path, "ensemble_predictions")
        evaluator.save_predictions(ensemble_predictions, None, output_dir)
        
        return fold_results
    
    def hyperparameter_search(self, data_path: str, model_path: str, 
                            param_grid: dict) -> dict:
        """Perform hyperparameter search"""
        print("Starting hyperparameter search...")
        
        best_score = -np.inf
        best_params = None
        results = []
        
        # Simple grid search implementation
        for lr in param_grid.get('learning_rate', [1e-5]):
            for batch_size in param_grid.get('batch_size', [2]):
                for epochs in param_grid.get('num_epochs', [6]):
                    
                    # Update config
                    self.config.learning_rate = lr
                    self.config.batch_size = batch_size
                    self.config.num_epochs = epochs
                    
                    print(f"\nTesting: lr={lr}, batch_size={batch_size}, epochs={epochs}")
                    
                    # Train one fold for quick evaluation
                    trainer = Trainer(self.config)
                    val_loss = trainer.train_fold(1, data_path, model_path)
                    
                    # Convert loss to score (negative loss)
                    score = -val_loss
                    
                    results.append({
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'num_epochs': epochs,
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'num_epochs': epochs
                        }
                    
                    print(f"Score: {score:.4f}")
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def create_model_comparison(self, data_path: str, model_path: str, 
                              model_types: List[str]) -> dict:
        """Compare different model architectures"""
        print("Comparing model architectures...")
        
        results = {}
        
        for model_type in model_types:
            print(f"\nEvaluating {model_type}...")
            
            # Update config for this model
            temp_config = Config()
            temp_config.__dict__.update(self.config.__dict__)
            
            # Train and evaluate
            trainer = Trainer(temp_config, model_type)
            fold_results = trainer.train_all_folds(data_path, model_path)
            
            evaluator = Evaluator(temp_config, model_type)
            cv_results = evaluator.evaluate_all_folds(data_path, model_path)
            
            results[model_type] = {
                'training_loss': fold_results,
                'cv_correlation': cv_results,
                'mean_cv_correlation': np.mean(cv_results)
            }
        
        # Print comparison
        print(f"\n{'='*60}")
        print("Model Architecture Comparison")
        print(f"{'='*60}")
        print(f"{'Model Type':<25} {'Mean CV Correlation':<20}")
        print(f"{'-'*60}")
        
        for model_type, result in results.items():
            print(f"{model_type:<25} {result['mean_cv_correlation']:<20.4f}")
        
        return results
