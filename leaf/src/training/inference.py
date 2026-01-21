import torch
import numpy as np
from tqdm import tqdm
from ..utils.config import Config

def predict(model, data_loader, device=Config.DEVICE):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for sample in tqdm(data_loader, desc="Predicting"):
            image = sample[0].float().to(device)
            preds = model(image)
            predictions.append(preds.cpu().numpy())
    
    predictions = np.vstack(predictions)
    return predictions

def predict_with_tta(model, data_loader, device=Config.DEVICE, num_tta=5):
    model.eval()
    predictions = []
    
    for tta_idx in range(num_tta):
        tta_predictions = []
        
        with torch.no_grad():
            for sample in tqdm(data_loader, desc=f"TTA {tta_idx+1}/{num_tta}"):
                image = sample[0].float().to(device)
                preds = model(image)
                tta_predictions.append(preds.cpu().numpy())
        
        tta_predictions = np.vstack(tta_predictions)
        predictions.append(tta_predictions)
    
    final_predictions = np.mean(predictions, axis=0)
    return final_predictions
