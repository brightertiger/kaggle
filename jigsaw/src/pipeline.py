#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from .config import Config
from .data_utils import DataProcessor, DataLoader
from .models import BERTClassifier, GPTClassifier, ModelTrainer, CustomLoss
from .evaluation import ModelEvaluator, BiasEvaluator


class JigsawPipeline:
    """Main pipeline for Jigsaw Toxic Comment Classification."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.data_loader = DataLoader(config)
        self.evaluator = ModelEvaluator(config)
        
        self.models = {}
        self.predictions = {}
        
        self._setup_device()
    
    def _setup_device(self):
        """Setup computing device."""
        if torch.cuda.is_available() and self.config.device.startswith('cuda'):
            self.device = torch.device(self.config.device)
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
    
    def validate_setup(self) -> bool:
        """Validate that all required files and dependencies are available."""
        print("Validating setup...")
        
        if not self.config.validate_setup():
            return False
        
        try:
            import transformers
            print(f"✅ Transformers version: {transformers.__version__}")
        except ImportError:
            print("❌ Transformers library not found")
            return False
        
        try:
            import torch
            print(f"✅ PyTorch version: {torch.__version__}")
        except ImportError:
            print("❌ PyTorch library not found")
            return False
        
        self.config.create_directories()
        print("✅ Setup validation completed")
        return True
    
    def process_data(self) -> Dict[str, Any]:
        """Process data and create training splits."""
        print("Processing data...")
        return self.data_processor.process_all_data()
    
    def get_tokenizer(self, model_type: str = 'bert'):
        """Get appropriate tokenizer for the model type."""
        try:
            from transformers import BertTokenizer, GPT2Tokenizer
            
            if model_type == 'bert':
                tokenizer = BertTokenizer.from_pretrained(
                    self.config.bert_config['model_name'],
                    do_lower_case=True
                )
            else:
                tokenizer = GPT2Tokenizer.from_pretrained(
                    self.config.gpt_config['model_name']
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer
        except ImportError:
            from pytorch_pretrained_bert import BertTokenizer, GPT2Tokenizer
            
            if model_type == 'bert':
                tokenizer = BertTokenizer.from_pretrained(
                    self.config.bert_config['model_name'],
                    do_lower_case=True
                )
            else:
                tokenizer = GPT2Tokenizer.from_pretrained(
                    self.config.gpt_config['model_name']
                )
            
            return tokenizer
    
    def train_bert_model(self, fold: int) -> Dict[str, Any]:
        """Train BERT model for a specific fold."""
        print(f"Training BERT model for fold {fold}...")
        
        train_data, valid_data = self.data_loader.load_fold_data(fold)
        tokenizer = self.get_tokenizer('bert')
        
        model = BERTClassifier(self.config).to(self.device)
        trainer = ModelTrainer(self.config, 'bert')
        
        train_loader, valid_loader = trainer.create_data_loaders(
            train_data, valid_data, tokenizer
        )
        
        num_training_steps = len(train_loader) * self.config.bert_config['num_epochs']
        optimizer = trainer.setup_optimizer(model, num_training_steps)
        
        loss_fn = CustomLoss.bert_loss
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.bert_config['num_epochs']):
            print(f"Epoch {epoch + 1}/{self.config.bert_config['num_epochs']}")
            
            train_loss = trainer.train_epoch(
                model, train_loader, optimizer, loss_fn, self.device
            )
            
            valid_loss, valid_predictions = trainer.validate_model(
                model, valid_loader, self.device
            )
            
            print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                patience_counter = 0
                
                model_path = self.config.get_model_path(
                    'bert', f'fold_{fold}_best_model.pt'
                )
                trainer.save_model(model, optimizer, epoch, valid_loss, model_path)
                
                predictions_path = self.config.get_output_path(
                    'predictions', f'bert_fold_{fold}_predictions.csv'
                )
                valid_predictions.to_csv(predictions_path, index=False)
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.training_config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        return {
            'model': model,
            'best_loss': best_loss,
            'predictions': valid_predictions
        }
    
    def train_gpt_model(self, fold: int) -> Dict[str, Any]:
        """Train GPT model for a specific fold."""
        print(f"Training GPT model for fold {fold}...")
        
        train_data, valid_data = self.data_loader.load_fold_data(fold)
        tokenizer = self.get_tokenizer('gpt')
        
        model = GPTClassifier(self.config).to(self.device)
        trainer = ModelTrainer(self.config, 'gpt')
        
        train_loader, valid_loader = trainer.create_data_loaders(
            train_data, valid_data, tokenizer
        )
        
        num_training_steps = len(train_loader) * self.config.gpt_config['num_epochs']
        optimizer = trainer.setup_optimizer(model, num_training_steps)
        
        loss_fn = CustomLoss.gpt_loss
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.gpt_config['num_epochs']):
            print(f"Epoch {epoch + 1}/{self.config.gpt_config['num_epochs']}")
            
            train_loss = trainer.train_epoch(
                model, train_loader, optimizer, loss_fn, self.device
            )
            
            valid_loss, valid_predictions = trainer.validate_model(
                model, valid_loader, self.device
            )
            
            print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                patience_counter = 0
                
                model_path = self.config.get_model_path(
                    'gpt', f'fold_{fold}_best_model.pt'
                )
                trainer.save_model(model, optimizer, epoch, valid_loss, model_path)
                
                predictions_path = self.config.get_output_path(
                    'predictions', f'gpt_fold_{fold}_predictions.csv'
                )
                valid_predictions.to_csv(predictions_path, index=False)
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.training_config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        return {
            'model': model,
            'best_loss': best_loss,
            'predictions': valid_predictions
        }
    
    def train_all_models(self, model_type: str = 'both') -> Dict[str, Any]:
        """Train models for all folds."""
        print(f"Training {model_type} models for all folds...")
        
        results = {}
        
        for fold in range(1, self.config.n_folds + 1):
            print(f"\n{'='*50}")
            print(f"Training Fold {fold}/{self.config.n_folds}")
            print(f"{'='*50}")
            
            fold_results = {}
            
            if model_type in ['bert', 'both']:
                fold_results['bert'] = self.train_bert_model(fold)
            
            if model_type in ['gpt', 'both']:
                fold_results['gpt'] = self.train_gpt_model(fold)
            
            results[f'fold_{fold}'] = fold_results
        
        self.models = results
        return results
    
    def evaluate_models(self) -> Dict[str, Any]:
        """Evaluate all trained models."""
        print("Evaluating models...")
        
        evaluation_results = {}
        
        for fold_name, fold_results in self.models.items():
            print(f"\nEvaluating {fold_name}...")
            
            fold_evaluations = {}
            
            for model_name, model_results in fold_results.items():
                predictions = model_results['predictions']
                
                valid_data, _ = self.data_loader.load_fold_data(int(fold_name.split('_')[1]))
                ground_truth = valid_data[['id', 'target'] + self.config.identity_columns]
                
                eval_path = self.config.get_output_path(
                    'evaluations', f'{model_name}_{fold_name}_evaluation'
                )
                
                evaluation = self.evaluator.evaluate_predictions(
                    predictions, ground_truth, eval_path
                )
                
                fold_evaluations[model_name] = evaluation
                print(f"{model_name} - Final Metric: {evaluation['final_metric']:.5f}")
            
            evaluation_results[fold_name] = fold_evaluations
        
        return evaluation_results
    
    def generate_test_predictions(self, model_type: str = 'bert') -> pd.DataFrame:
        """Generate predictions on test data."""
        print(f"Generating test predictions with {model_type} models...")
        
        test_data = self.data_loader.load_test_data()
        tokenizer = self.get_tokenizer(model_type)
        
        all_predictions = []
        
        for fold in range(1, self.config.n_folds + 1):
            print(f"Generating predictions for fold {fold}...")
            
            if model_type == 'bert':
                model = BERTClassifier(self.config).to(self.device)
            else:
                model = GPTClassifier(self.config).to(self.device)
            
            model_path = self.config.get_model_path(
                model_type, f'fold_{fold}_best_model.pt'
            )
            
            if os.path.exists(model_path):
                trainer = ModelTrainer(self.config, model_type)
                model = trainer.load_model(model, model_path)
                
                model.eval()
                fold_predictions = []
                
                with torch.no_grad():
                    for _, row in test_data.iterrows():
                        text = str(row['comment_text'])
                        
                        if model_type == 'bert':
                            token_ids = self._tokenize_bert_text(text, tokenizer)
                        else:
                            token_ids = self._tokenize_gpt_text(text, tokenizer)
                        
                        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
                        
                        if model_type == 'bert':
                            prediction, _ = model(input_ids)
                        else:
                            prediction = model(input_ids)
                        
                        prediction = torch.sigmoid(prediction).cpu().item()
                        fold_predictions.append(prediction)
                
                all_predictions.append(fold_predictions)
            else:
                print(f"Model not found for fold {fold}: {model_path}")
        
        if all_predictions:
            avg_predictions = np.mean(all_predictions, axis=0)
            
            submission = pd.DataFrame({
                'id': test_data['id'],
                'prediction': avg_predictions
            })
            
            submission_path = self.config.get_output_path(
                'submissions', f'{model_type}_submission.csv'
            )
            submission.to_csv(submission_path, index=False)
            print(f"Submission saved to: {submission_path}")
            
            return submission
        else:
            print("No predictions generated")
            return pd.DataFrame()
    
    def _tokenize_bert_text(self, text: str, tokenizer) -> List[int]:
        """Tokenize text for BERT model."""
        if hasattr(tokenizer, 'encode_plus'):
            encoding = tokenizer.encode_plus(
                text,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            return encoding['input_ids'].flatten().tolist()
        else:
            tokens = tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens[:self.config.max_length-2] + ["[SEP]"]
            return tokenizer.convert_tokens_to_ids(tokens)
    
    def _tokenize_gpt_text(self, text: str, tokenizer) -> List[int]:
        """Tokenize text for GPT model."""
        if hasattr(tokenizer, 'encode_plus'):
            encoding = tokenizer.encode_plus(
                text,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            return encoding['input_ids'].flatten().tolist()
        else:
            return tokenizer.encode(text, max_length=self.config.max_length, truncation=True)
    
    def run_full_pipeline(self, model_type: str = 'both') -> Dict[str, Any]:
        """Run the complete pipeline."""
        print("Starting full pipeline...")
        
        if not self.validate_setup():
            raise RuntimeError("Setup validation failed")
        
        self.process_data()
        self.train_all_models(model_type)
        evaluation_results = self.evaluate_models()
        
        test_predictions = {}
        if 'bert' in model_type or model_type == 'both':
            test_predictions['bert'] = self.generate_test_predictions('bert')
        
        if 'gpt' in model_type or model_type == 'both':
            test_predictions['gpt'] = self.generate_test_predictions('gpt')
        
        return {
            'models': self.models,
            'evaluations': evaluation_results,
            'test_predictions': test_predictions
        }
