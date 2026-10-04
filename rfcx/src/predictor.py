import torch
import torch.nn as nn
from dataclasses import replace
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional
from .config import Config
from .models import create_model
from .data_utils import create_test_loader

class ModelPredictor:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

    def predict_fold(self, model: nn.Module, test_loader: DataLoader) -> np.ndarray:
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for sample in test_loader:
                sound = sample.float().squeeze(0).to(self.device)
                # Bound inference memory while retaining maximum evidence per class.
                chunk_predictions = [model(chunk).amax(dim=0) for chunk in
                                     sound.split(self.config.training.batch_size)]
                preds = torch.stack(chunk_predictions).amax(dim=0)
                predictions.append(preds.cpu().numpy())
        
        return np.vstack(predictions)

    def predict_all_folds(self, model_type: str = "resnet", apply_tta: bool = False) -> pd.DataFrame:
        test_loader = create_test_loader(self.config, apply_tta=apply_tta)
        test_data = pd.read_csv(self.config.data.test_data_path)
        
        all_predictions = []
        
        for fold in range(1, self.config.training.num_folds + 1):
            print(f"Predicting fold {fold}...")
            
            inference_config = replace(self.config, model=replace(self.config.model, pretrained=False))
            model = create_model(inference_config, model_type)
            model = model.to(self.device)
            
            model_path = f"{self.config.data.model_save_path}/{model_type}/model_fold_{fold}.pt"
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            fold_predictions = self.predict_fold(model, test_loader)
            all_predictions.append(fold_predictions)
            
            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        predictions = torch.from_numpy(np.mean(all_predictions, axis=0)).sigmoid().numpy()
        
        result_df = test_data[['recording_id']].copy()
        prediction_columns = [f's{i}' for i in range(self.config.model.num_classes)]
        result_df[prediction_columns] = predictions
        
        return result_df

def generate_predictions(config: Config, model_type: str = "resnet", 
                        apply_tta: bool = False, output_name: Optional[str] = None) -> None:
    predictor = ModelPredictor(config)
    predictions = predictor.predict_all_folds(model_type, apply_tta)
    
    if output_name is None:
        output_name = f"{model_type}_predictions"
        if apply_tta:
            output_name += "_tta"
    
    output_path = f"{config.data.predictions_path}/{output_name}.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
