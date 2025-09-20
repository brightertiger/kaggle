import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data_utils import MelanomaDataset
from .config import Config

class MelanomaInference:
    def __init__(self, model, device=Config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, test_loader, use_tta=True):
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Inference'):
                images = batch['image'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                
                if use_tta:
                    # Test Time Augmentation
                    predictions = []
                    
                    # Original
                    outputs = self.model(images, metadata)
                    predictions.append(torch.sigmoid(outputs))
                    
                    # Horizontal flip
                    images_flip = torch.flip(images, dims=[3])
                    outputs_flip = self.model(images_flip, metadata)
                    predictions.append(torch.sigmoid(outputs_flip))
                    
                    # Vertical flip
                    images_vflip = torch.flip(images, dims=[2])
                    outputs_vflip = self.model(images_vflip, metadata)
                    predictions.append(torch.sigmoid(outputs_vflip))
                    
                    # Both flips
                    images_both = torch.flip(torch.flip(images, dims=[3]), dims=[2])
                    outputs_both = self.model(images_both, metadata)
                    predictions.append(torch.sigmoid(outputs_both))
                    
                    # Average predictions
                    final_prediction = torch.stack(predictions).mean(dim=0)
                else:
                    outputs = self.model(images, metadata)
                    final_prediction = torch.sigmoid(outputs)
                
                all_predictions.append(final_prediction.cpu().numpy())
        
        return np.vstack(all_predictions)
    
    def predict_single_fold(self, test_dataset, batch_size=Config.BATCH_SIZE, num_workers=3, use_tta=True):
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        predictions = self.predict(test_loader, use_tta)
        
        # Extract melanoma probability (class 1)
        melanoma_probs = predictions[:, 1]
        
        return melanoma_probs
    
    def predict_multiple_models(self, models, test_dataset, batch_size=Config.BATCH_SIZE, num_workers=3, use_tta=True):
        all_predictions = []
        
        for i, model in enumerate(models):
            print(f'Predicting with model {i+1}/{len(models)}')
            self.model = model.to(self.device)
            self.model.eval()
            
            predictions = self.predict_single_fold(test_dataset, batch_size, num_workers, use_tta)
            all_predictions.append(predictions)
        
        # Average predictions from all models
        ensemble_predictions = np.mean(all_predictions, axis=0)
        
        return ensemble_predictions

def create_test_dataset(image_path, metadata_df):
    return MelanomaDataset(image_path, metadata_df, fold=None, is_training=False)

def load_trained_model(model_path, model_class, device=Config.DEVICE):
    model = model_class()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
