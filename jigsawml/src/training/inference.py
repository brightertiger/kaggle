import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..utils.config import Config

class ModelInference:
    def __init__(self, model, device=Config.DEVICE):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict(self, data_loader):
        scores = []
        ids = []
        
        tq = tqdm(total=len(data_loader) * 24, disable=False)
        
        with torch.no_grad():
            for batch in data_loader:
                batch_ids = batch.pop('id')
                
                for key, value in batch.items():
                    batch[key] = value.to(self.device)
                
                predictions = torch.sigmoid(self.model(**batch))
                
                scores.append(predictions.cpu().data.numpy().reshape(-1, 1))
                ids.append(batch_ids.data.numpy().reshape(-1, 1))
                
                tq.update(24)
        
        tq.close()
        
        scores = np.vstack(scores)
        ids = np.vstack(ids)
        
        results = pd.DataFrame({
            'id': ids.flatten(),
            'toxic': scores.flatten()
        })
        
        return results
