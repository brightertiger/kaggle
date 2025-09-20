import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool
from functools import reduce
from src.config import Config
from src.data_utils import DataProcessor, TextPreprocessor
from src.models import NeuralNetworkModel, NaiveBayesSVM, LogisticRegressionModel
from src.ensemble import EnsembleModel, ModelEvaluator


class ToxicCommentPipeline:
    """Main pipeline for toxic comment classification"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.ensemble_model = EnsembleModel(config)
        self.evaluator = ModelEvaluator(config)
        
        # Create necessary directories
        self.config.create_directories()
    
    def prepare_data(self):
        """Prepare all data with different preprocessing methods"""
        print("Preparing data with multiple preprocessing methods...")
        self.data_processor.process_all_data()
        print("Data preparation completed!")
    
    def train_neural_network_models(self, embedding_type: str = "glove") -> Dict[str, List[pd.DataFrame]]:
        """Train neural network models with different preprocessing methods"""
        print(f"Training neural network models with {embedding_type} embeddings...")
        
        neural_model = NeuralNetworkModel(self.config)
        
        # Load embeddings
        if embedding_type == "glove":
            embedding_path = self.config.model.glove_path
        else:
            embedding_path = self.config.model.fasttext_path
        
        embeddings_index = neural_model.load_embeddings(embedding_path)
        
        results = {}
        
        for method in self.config.data.preprocessing_methods[:2]:  # Use first 2 methods
            print(f"Training neural models for preprocessing method: {method}")
            results[method] = self._train_nn_for_preprocessing(
                neural_model, embeddings_index, method, embedding_type
            )
        
        return results
    
    def _train_nn_for_preprocessing(self, neural_model: NeuralNetworkModel, 
                                  embeddings_index: Dict[str, np.ndarray],
                                  preprocessing_method: str,
                                  embedding_type: str) -> List[pd.DataFrame]:
        """Train neural network models for a specific preprocessing method"""
        results = []
        
        for fold in range(1, self.config.data.n_folds + 1):
            print(f"Training fold {fold} for {preprocessing_method}")
            
            # Load data
            train_text = pd.read_csv(f"{self.config.data.output_dir}/{preprocessing_method}/train/train_data_{fold}.csv")
            train_label = pd.read_csv(f"{self.config.data.output_dir}/{preprocessing_method}/train/train_labels_{fold}.csv")
            valid_text = pd.read_csv(f"{self.config.data.output_dir}/{preprocessing_method}/train/test_data_{fold}.csv")
            test_text = pd.read_csv(f"{self.config.data.output_dir}/{preprocessing_method}/score/score_data.csv")
            
            # Prepare data
            train_seq, valid_seq, word_index = neural_model.prepare_data(train_text, valid_text)
            embedding_matrix = neural_model.create_embedding_matrix(
                neural_model.tokenizer, embeddings_index
            )
            
            # Build and train model
            model = neural_model.build_model(embedding_matrix)
            
            # Training parameters
            train_params = {
                'x': train_seq,
                'y': np.array(train_label.iloc[:, 1:]),
                'validation_data': (valid_seq, np.array(valid_text.iloc[:, 1:])),
                'batch_size': self.config.model.batch_size,
                'epochs': self.config.model.epochs,
                'verbose': 0
            }
            
            # Train model
            model.fit(**train_params)
            
            # Make predictions
            valid_predictions = model.predict(valid_seq)
            test_predictions = model.predict(neural_model.tokenizer.texts_to_sequences(
                test_text['comment_text'].fillna('nan')
            ))
            
            # Create result DataFrames
            valid_result = valid_text[['id']].copy()
            test_result = test_text[['id']].copy()
            
            for i, col in enumerate(self.config.evaluation.target_columns):
                valid_result[col] = valid_predictions[:, i]
                test_result[col] = test_predictions[:, i]
            
            results.append((valid_result, test_result))
        
        return results
    
    def train_traditional_models(self) -> Dict[str, pd.DataFrame]:
        """Train traditional ML models"""
        print("Training traditional ML models...")
        
        results = {}
        
        # Train NB-SVM
        print("Training NB-SVM model...")
        nb_svm = NaiveBayesSVM(self.config)
        nb_svm_results = self._train_traditional_model(nb_svm, "nbsvm")
        results['nbsvm'] = nb_svm_results
        
        # Train Logistic Regression
        print("Training Logistic Regression model...")
        lr_model = LogisticRegressionModel(self.config)
        lr_results = self._train_traditional_model(lr_model, "logistic_regression")
        results['logistic_regression'] = lr_results
        
        return results
    
    def _train_traditional_model(self, model, model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train a traditional model across all folds"""
        valid_scores = []
        test_scores = []
        
        for fold in range(1, self.config.data.n_folds + 1):
            print(f"Training {model_name} fold {fold}")
            
            # Load data
            train_text = pd.read_csv(f"{self.config.data.output_dir}/source_6/train/train_data_{fold}.csv")
            train_label = pd.read_csv(f"{self.config.data.output_dir}/source_6/train/train_labels_{fold}.csv")
            valid_text = pd.read_csv(f"{self.config.data.output_dir}/source_6/train/test_data_{fold}.csv")
            test_text = pd.read_csv(f"{self.config.data.output_dir}/source_6/score/score_data.csv")
            
            # Train model
            valid_score, test_score = model.train_model(train_label, train_text, valid_text, test_text)
            
            valid_scores.append(valid_score)
            test_scores.append(test_score)
        
        # Combine results
        valid_data = reduce(lambda x, y: pd.concat([x, y]), valid_scores)
        test_data = reduce(lambda x, y: pd.concat([x, y]), test_scores)
        test_data = test_data.groupby('id').mean().reset_index()
        
        return valid_data, test_data
    
    def create_ensemble_predictions(self, model_predictions: Dict[str, pd.DataFrame],
                                  weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Create ensemble predictions from multiple models"""
        print("Creating ensemble predictions...")
        
        if weights is None:
            # Equal weights
            weights = {model_name: 1.0 for model_name in model_predictions.keys()}
        
        ensemble_pred = self.ensemble_model.blend_predictions(
            model_predictions, weights, self.config.evaluation.target_columns
        )
        
        return ensemble_pred
    
    def evaluate_models(self, predictions: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Evaluate all model predictions"""
        print("Evaluating model performance...")
        
        # Load ground truth labels
        train_labels = pd.read_csv(self.config.data.train_path)
        train_labels = train_labels.drop('comment_text', axis=1)
        
        # Filter labels to match predictions
        for model_name, pred in predictions.items():
            if 'id' in pred.columns:
                filtered_labels = train_labels.merge(pred[['id']], on='id')
                results = self.evaluator.evaluate_single_model(
                    pred, filtered_labels, self.config.evaluation.target_columns
                )
                predictions[f"{model_name}_eval"] = results
        
        # Get evaluation results
        eval_results = {}
        for key, value in predictions.items():
            if key.endswith('_eval'):
                eval_results[key.replace('_eval', '')] = value
        
        return eval_results
    
    def save_predictions(self, predictions: Dict[str, pd.DataFrame]):
        """Save model predictions"""
        print("Saving predictions...")
        
        for model_name, pred in predictions.items():
            if isinstance(pred, pd.DataFrame):
                # Save validation predictions
                val_path = f"{self.config.model.model_dir}/{model_name}_validation.csv"
                pred.to_csv(val_path, index=False)
                
                # Save test predictions (if available)
                if 'id' in pred.columns:
                    test_path = f"submissions/{model_name}_submission.csv"
                    pred.to_csv(test_path, index=False)
    
    def run_full_pipeline(self, train_neural: bool = True, train_traditional: bool = True):
        """Run the complete pipeline"""
        print("Starting Toxic Comment Classification Pipeline...")
        
        # Step 1: Prepare data
        self.prepare_data()
        
        all_predictions = {}
        
        # Step 2: Train neural network models
        if train_neural:
            nn_results = self.train_neural_network_models("glove")
            for method, results in nn_results.items():
                # Average predictions across folds
                valid_preds = [result[0] for result in results]
                test_preds = [result[1] for result in results]
                
                valid_ensemble = self.ensemble_model.simple_averaging(
                    valid_preds, self.config.evaluation.target_columns
                )
                test_ensemble = self.ensemble_model.simple_averaging(
                    test_preds, self.config.evaluation.target_columns
                )
                
                all_predictions[f"neural_{method}"] = valid_ensemble
                all_predictions[f"neural_{method}_test"] = test_ensemble
        
        # Step 3: Train traditional models
        if train_traditional:
            traditional_results = self.train_traditional_models()
            for model_name, (valid_pred, test_pred) in traditional_results.items():
                all_predictions[model_name] = valid_pred
                all_predictions[f"{model_name}_test"] = test_pred
        
        # Step 4: Create final ensemble
        final_ensemble = self.create_ensemble_predictions(all_predictions)
        all_predictions['final_ensemble'] = final_ensemble
        
        # Step 5: Evaluate models
        eval_results = self.evaluate_models(all_predictions)
        self.evaluator.print_evaluation_results(eval_results)
        
        # Step 6: Save predictions
        self.save_predictions(all_predictions)
        
        print("Pipeline completed successfully!")
        return all_predictions, eval_results
