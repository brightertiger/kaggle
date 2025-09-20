import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..core import Config
from ..models import create_model
from ..data import create_data_loaders
from .predictor import ModelPredictor, load_trained_model

class ModelValidator:
    """Validation utilities for trained models"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def validate_fold(self, fold_idx: int, model_name: str = 'resnext101') -> pd.DataFrame:
        """Validate model on a specific fold"""
        
        print(f"Validating fold {fold_idx} with {model_name}")
        
        # Load trained model
        model_path = self.config.MODEL_DIR / f"model_{fold_idx}.pt"
        if not model_path.exists():
            model_path = self.config.MODEL_DIR / "best_model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model found for fold {fold_idx}")
        
        model = load_trained_model(str(model_path), model_name, self.config)
        predictor = ModelPredictor(model, self.config.DEVICE, self.config)
        
        # Get validation data
        _, valid_loader = create_data_loaders(fold_idx, self.config)
        
        # Generate predictions
        results = predictor.predict_loader(valid_loader, apply_sigmoid=True)
        
        # Create validation dataframe
        validation_df = pd.DataFrame(
            results['predictions'], 
            columns=self.config.CLASS_NAMES
        )
        validation_df['image'] = results['indices']
        validation_df['fold'] = fold_idx
        
        # Load ground truth labels
        train_data = pd.read_csv(self.config.TRAIN_CSV)
        train_data = train_data[train_data['fold'] == fold_idx]
        
        # Merge with ground truth
        validation_df = validation_df.merge(
            train_data[['image'] + self.config.CLASS_NAMES], 
            on='image', 
            suffixes=('_pred', '_true')
        )
        
        model = model.cpu()
        del model, predictor
        torch.cuda.empty_cache()
        
        return validation_df
    
    def validate_all_folds(self, model_name: str = 'resnext101') -> List[pd.DataFrame]:
        """Validate all folds"""
        
        print("Validating all folds...")
        
        all_validations = []
        
        for fold_idx in range(1, self.config.NUM_FOLDS + 1):
            try:
                fold_validation = self.validate_fold(fold_idx, model_name)
                all_validations.append(fold_validation)
                
                # Save individual fold validation
                validation_path = self.config.OUTPUT_DIR / f"validation_fold_{fold_idx}.csv"
                fold_validation.to_csv(validation_path, index=False)
                
                print(f"Fold {fold_idx} validation completed")
                
            except Exception as e:
                print(f"Error validating fold {fold_idx}: {e}")
        
        return all_validations
    
    def calculate_metrics(self, validation_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate validation metrics"""
        
        metrics = {}
        
        for class_name in self.config.CLASS_NAMES:
            pred_col = f"{class_name}_pred"
            true_col = f"{class_name}_true"
            
            if pred_col in validation_df.columns and true_col in validation_df.columns:
                # Convert predictions to binary
                y_pred = (validation_df[pred_col] > 0.5).astype(int)
                y_true = validation_df[true_col].astype(int)
                
                # Calculate metrics
                accuracy = (y_pred == y_true).mean()
                precision = self._calculate_precision(y_true, y_pred)
                recall = self._calculate_recall(y_true, y_pred)
                f1 = self._calculate_f1(y_true, y_pred)
                
                metrics[f"{class_name}_accuracy"] = accuracy
                metrics[f"{class_name}_precision"] = precision
                metrics[f"{class_name}_recall"] = recall
                metrics[f"{class_name}_f1"] = f1
        
        return metrics
    
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision"""
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall"""
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score"""
        precision = self._calculate_precision(y_true, y_pred)
        recall = self._calculate_recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def generate_validation_report(self, all_validations: List[pd.DataFrame]) -> str:
        """Generate comprehensive validation report"""
        
        report = []
        report.append("=" * 60)
        report.append("MODEL VALIDATION REPORT")
        report.append("=" * 60)
        
        # Combine all validations
        combined_validation = pd.concat(all_validations, ignore_index=True)
        
        # Overall metrics
        overall_metrics = self.calculate_metrics(combined_validation)
        
        report.append(f"\nOverall Performance:")
        report.append(f"- Total validation samples: {len(combined_validation):,}")
        report.append(f"- Number of folds: {len(all_validations)}")
        
        report.append(f"\nClass-wise Performance:")
        for class_name in self.config.CLASS_NAMES:
            accuracy = overall_metrics.get(f"{class_name}_accuracy", 0)
            precision = overall_metrics.get(f"{class_name}_precision", 0)
            recall = overall_metrics.get(f"{class_name}_recall", 0)
            f1 = overall_metrics.get(f"{class_name}_f1", 0)
            
            report.append(f"\n{class_name.upper()}:")
            report.append(f"  - Accuracy:  {accuracy:.4f}")
            report.append(f"  - Precision: {precision:.4f}")
            report.append(f"  - Recall:    {recall:.4f}")
            report.append(f"  - F1 Score:  {f1:.4f}")
        
        # Per-fold metrics
        report.append(f"\nPer-fold Performance:")
        for i, fold_validation in enumerate(all_validations, 1):
            fold_metrics = self.calculate_metrics(fold_validation)
            
            report.append(f"\nFold {i}:")
            report.append(f"  - Samples: {len(fold_validation):,}")
            
            for class_name in self.config.CLASS_NAMES:
                f1 = fold_metrics.get(f"{class_name}_f1", 0)
                report.append(f"  - {class_name} F1: {f1:.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

def run_validation(config: Config, model_name: str = 'resnext101') -> None:
    """Run complete validation pipeline"""
    
    print("Starting model validation...")
    
    validator = ModelValidator(config)
    
    # Validate all folds
    all_validations = validator.validate_all_folds(model_name)
    
    if not all_validations:
        print("No validations completed successfully")
        return
    
    # Generate report
    report = validator.generate_validation_report(all_validations)
    print(report)
    
    # Save report
    report_path = config.OUTPUT_DIR / "validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nValidation report saved to: {report_path}")
    
    # Save combined validation results
    combined_validation = pd.concat(all_validations, ignore_index=True)
    combined_path = config.OUTPUT_DIR / "combined_validation.csv"
    combined_validation.to_csv(combined_path, index=False)
    
    print(f"Combined validation results saved to: {combined_path}")
    print("Model validation completed!")

if __name__ == "__main__":
    config = Config()
    run_validation(config)
