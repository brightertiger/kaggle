import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch.utils.data import DataLoader
from functools import reduce

from ..core.config import Config
from ..models.models import create_model
from ..data.data_utils import create_test_loader, SaltScoreDataset

class ModelPredictor:
    """Model prediction class for Salt Identification"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
    def load_model(self, fold_idx: int, model_name: str) -> torch.nn.Module:
        """Load trained model for a specific fold"""
        model_path = self.config.MODEL_DIR / f"model_{fold_idx}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create and load model
        model = create_model(model_name, self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Loaded model for fold {fold_idx}")
        return model
    
    def predict_single_fold(self, fold_idx: int, model_name: str, 
                           use_tta: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for a single fold"""
        model = self.load_model(fold_idx, model_name)
        
        # Create data loaders
        valid_loader = self._create_validation_loader(fold_idx)
        test_loader = self._create_test_loader()
        
        # Validation predictions
        valid_scores, valid_actuals = self._predict_loader(model, valid_loader, use_tta)
        
        # Test predictions
        test_scores = self._predict_loader(model, test_loader, use_tta, has_masks=False)
        
        # Save predictions
        self._save_predictions(fold_idx, valid_scores, valid_actuals, test_scores)
        
        return valid_scores, test_scores
    
    def _create_validation_loader(self, fold_idx: int) -> DataLoader:
        """Create validation data loader"""
        from ..data.data_utils import create_data_loaders
        _, valid_loader = create_data_loaders(fold_idx, self.config)
        return valid_loader
    
    def _create_test_loader(self) -> DataLoader:
        """Create test data loader"""
        return create_test_loader(self.config)
    
    def _predict_loader(self, model: torch.nn.Module, data_loader: DataLoader, 
                       use_tta: bool = False, has_masks: bool = True) -> np.ndarray:
        """Generate predictions for a data loader"""
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_idx, sample in enumerate(data_loader):
                images = sample['image'].to(self.device)
                
                if use_tta:
                    # Test Time Augmentation
                    preds = self._predict_with_tta(model, images)
                else:
                    preds = model(images)
                
                predictions.append(preds.cpu().numpy())
                
                if has_masks and 'mask' in sample:
                    masks = sample['mask'].cpu().numpy()
                    actuals.append(masks)
        
        predictions = np.vstack(predictions)
        
        if has_masks and actuals:
            actuals = np.vstack(actuals)
            return predictions, actuals
        
        return predictions
    
    def _predict_with_tta(self, model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Predict with Test Time Augmentation"""
        # Original prediction
        pred1 = model(images)
        
        # Horizontal flip prediction
        pred2 = model(torch.flip(images, dims=[3]))
        pred2 = torch.flip(pred2, dims=[3])
        
        # Average predictions
        return 0.5 * pred1 + 0.5 * pred2
    
    def _save_predictions(self, fold_idx: int, valid_scores: np.ndarray, 
                         valid_actuals: np.ndarray, test_scores: np.ndarray):
        """Save predictions to disk"""
        # Save validation predictions
        np.save(self.config.SCORES_DIR / "valid" / f"scores_{fold_idx}.npy", valid_scores)
        np.save(self.config.SCORES_DIR / "valid" / f"actuals_{fold_idx}.npy", valid_actuals)
        
        # Save test predictions
        np.save(self.config.SCORES_DIR / "test" / f"scores_{fold_idx}.npy", test_scores)
        
        print(f"Saved predictions for fold {fold_idx}")
    
    def predict_all_folds(self, model_name: str, use_tta: bool = False) -> Dict[int, np.ndarray]:
        """Generate predictions for all folds"""
        all_test_predictions = {}
        
        for fold_idx in range(1, self.config.NUM_FOLDS + 1):
            print(f"\nGenerating predictions for fold {fold_idx}...")
            valid_scores, test_scores = self.predict_single_fold(
                fold_idx, model_name, use_tta
            )
            all_test_predictions[fold_idx] = test_scores
        
        return all_test_predictions
    
    def create_submission(self, model_name: str, use_tta: bool = False, 
                         threshold: Optional[float] = None) -> pd.DataFrame:
        """Create submission file from ensemble predictions"""
        if threshold is None:
            threshold = self.config.IOU_CUTOFF
        
        # Load test indices
        test_path = self.config.PROCESSED_DATA_DIR / "test"
        import pickle
        with open(test_path / "test.pkl", 'rb') as f:
            test_indices = pickle.load(f)
        
        # Load all fold predictions
        all_predictions = []
        for fold_idx in range(1, self.config.NUM_FOLDS + 1):
            scores_path = self.config.SCORES_DIR / "test" / f"scores_{fold_idx}.npy"
            if scores_path.exists():
                scores = np.load(scores_path)
                all_predictions.append(scores)
        
        if not all_predictions:
            raise ValueError("No predictions found. Run prediction first.")
        
        # Ensemble predictions
        ensemble_predictions = reduce(lambda x, y: x + y, all_predictions)
        ensemble_predictions = ensemble_predictions / len(all_predictions)
        
        # Create submission
        submission_data = {}
        for idx, image_id in enumerate(test_indices):
            # Extract prediction for this image (remove padding)
            pred = ensemble_predictions[idx, 14:-13, 14:-13]
            pred = pred.reshape(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 1)
            
            # Apply threshold
            pred_binary = (pred >= threshold).astype(np.int32)
            
            # Filter small predictions
            if pred_binary.sum() <= self.config.MIN_SALT_PIXELS:
                pred_binary = np.zeros_like(pred_binary)
            
            # Convert to RLE
            rle = self._rle_encode(pred_binary)
            submission_data[image_id] = rle
        
        # Create DataFrame
        submission_df = pd.DataFrame.from_dict(
            list(submission_data.items()),
            columns=['id', 'rle_mask']
        )
        
        # Save submission
        submission_path = self.config.SUBMIT_DIR / f"{model_name}_submission.csv"
        submission_df.to_csv(submission_path, index=False)
        
        print(f"Submission saved to: {submission_path}")
        print(f"Total predictions: {len(submission_df)}")
        
        return submission_df
    
    def _rle_encode(self, image: np.ndarray) -> str:
        """Run Length Encoding for binary images"""
        pixels = image.flatten(order='F')
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
