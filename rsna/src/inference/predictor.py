import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path

from ..core import Config
from ..models import create_model

class ModelPredictor:
    """Prediction class for intracranial hemorrhage detection models"""
    
    def __init__(self, model: nn.Module, device: str, config: Config):
        self.model = model
        self.device = device
        self.config = config
    
    def predict_batch(self, batch: Dict[str, Any], apply_sigmoid: bool = True) -> np.ndarray:
        """Predict on a single batch"""
        self.model.eval()
        
        with torch.no_grad():
            images = batch['image'].float().to(self.device)
            predictions = self.model(images)
            
            if apply_sigmoid:
                predictions = torch.sigmoid(predictions)
            
            return predictions.cpu().numpy()
    
    def predict_loader(self, data_loader: torch.utils.data.DataLoader, 
                      apply_sigmoid: bool = True, 
                      return_indices: bool = True) -> Dict[str, Any]:
        """Predict on entire data loader"""
        self.model.eval()
        
        all_predictions = []
        all_indices = []
        
        progress_bar = tqdm(data_loader, desc="Predicting", leave=False)
        
        for batch in progress_bar:
            predictions = self.predict_batch(batch, apply_sigmoid)
            all_predictions.append(predictions)
            
            if return_indices:
                all_indices.extend(batch['idx'])
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        result = {'predictions': all_predictions}
        if return_indices:
            result['indices'] = all_indices
        
        return result
    
    def predict_with_tta(self, data_loader: torch.utils.data.DataLoader, 
                        tta_transforms: List[Any] = None) -> Dict[str, Any]:
        """Predict with Test Time Augmentation"""
        if tta_transforms is None:
            tta_transforms = [
                lambda x: x,  # Original
                lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
            ]
        
        all_tta_predictions = []
        
        for tta_idx, transform in enumerate(tta_transforms):
            print(f"TTA {tta_idx + 1}/{len(tta_transforms)}")
            
            tta_predictions = []
            tta_indices = []
            
            self.model.eval()
            progress_bar = tqdm(data_loader, desc=f"TTA {tta_idx + 1}", leave=False)
            
            for batch in progress_bar:
                images = transform(batch['image']).float().to(self.device)
                
                with torch.no_grad():
                    predictions = self.model(images)
                    predictions = torch.sigmoid(predictions)
                
                tta_predictions.append(predictions.cpu().numpy())
                if tta_idx == 0:
                    tta_indices.extend(batch['idx'])
            
            all_tta_predictions.append(np.concatenate(tta_predictions, axis=0))
        
        final_predictions = np.mean(all_tta_predictions, axis=0)
        
        return {
            'predictions': final_predictions,
            'indices': tta_indices,
            'tta_predictions': all_tta_predictions
        }

def load_trained_model(model_path: str, model_name: str, config: Config) -> nn.Module:
    """Load trained model from checkpoint"""
    model = create_model(model_name, config.NUM_CLASSES)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(config.DEVICE)
    
    try:
        from apex import amp
        model = amp.initialize(model, opt_level="O2", keep_batchnorm_fp32=True, verbosity=0)
    except ImportError:
        pass
    
    return model

def predict_fold(fold_idx: int, config: Config, model_name: str = 'resnext101') -> pd.DataFrame:
    """Generate predictions for a specific fold"""
    from ..data import create_inference_loader
    
    print(f"Generating predictions for fold {fold_idx}")
    
    model_path = config.MODEL_DIR / f"best_model.pt"
    if not model_path.exists():
        model_path = config.MODEL_DIR / f"model_{config.NUM_EPOCHS - 1}.pt"
    
    model = load_trained_model(str(model_path), model_name, config)
    predictor = ModelPredictor(model, config.DEVICE, config)
    
    inference_loader = create_inference_loader(config)
    
    results = predictor.predict_loader(inference_loader, apply_sigmoid=True)
    
    predictions_df = pd.DataFrame(
        results['predictions'], 
        columns=config.CLASS_NAMES
    )
    predictions_df['image'] = results['indices']
    
    predictions_df = predictions_df[['image'] + config.CLASS_NAMES]
    
    model = model.cpu()
    del model, predictor
    torch.cuda.empty_cache()
    
    return predictions_df

def create_submission(predictions_list: List[pd.DataFrame], config: Config) -> pd.DataFrame:
    """Create final submission from multiple fold predictions"""
    
    combined_predictions = pd.concat(predictions_list, ignore_index=True)
    
    ensemble_predictions = combined_predictions.groupby('image').mean().reset_index()
    
    submission_data = []
    
    for _, row in ensemble_predictions.iterrows():
        image_id = row['image']
        
        for class_name in config.CLASS_NAMES:
            submission_data.append({
                'ID': f'ID_{image_id}_{class_name}',
                'Label': row[class_name]
            })
    
    submission_df = pd.DataFrame(submission_data)
    
    return submission_df

def generate_all_predictions(config: Config, model_name: str = 'resnext101') -> pd.DataFrame:
    """Generate predictions for all folds and create ensemble submission"""
    
    all_predictions = []
    
    for fold_idx in range(1, config.NUM_FOLDS + 1):
        fold_predictions = predict_fold(fold_idx, config, model_name)
        all_predictions.append(fold_predictions)
        
        fold_predictions.to_csv(
            config.SCORE_DIR / f"predictions_fold_{fold_idx}.csv", 
            index=False
        )
    
    submission_df = create_submission(all_predictions, config)
    
    submission_df.to_csv(config.SCORE_DIR / "submission.csv", index=False)
    
    print(f"Submission saved with {len(submission_df)} predictions")
    print(f"Label statistics:")
    print(submission_df['Label'].describe())
    
    return submission_df
