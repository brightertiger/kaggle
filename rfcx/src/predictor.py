import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Optional
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
                sound = sample.float().squeeze().to(self.device)
                preds = model(sound)
                preds, _ = torch.max(preds, dim=0)
                predictions.append(preds.cpu().data.numpy())
        
        return np.vstack(predictions)

    def predict_all_folds(self, model_type: str = "resnet", apply_tta: bool = False) -> pd.DataFrame:
        test_loader = create_test_loader(self.config, apply_tta=apply_tta)
        test_data = pd.read_csv(self.config.data.test_data_path)
        
        all_predictions = []
        
        for fold in range(1, self.config.training.num_folds + 1):
            print(f"Predicting fold {fold}...")
            
            model = create_model(self.config, model_type)
            model = model.to(self.device)
            
            model_path = f"{self.config.data.model_save_path}/model_fold_{fold}.pt"
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            fold_predictions = self.predict_fold(model, test_loader)
            all_predictions.append(fold_predictions)
            
            model.cpu()
            del model
            torch.cuda.empty_cache()
        
        predictions = np.mean(all_predictions, axis=0)
        
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
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
