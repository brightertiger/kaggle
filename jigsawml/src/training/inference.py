import torch
import pandas as pd
from tqdm import tqdm
from ..utils.config import Config


class ModelInference:
    def __init__(self, model, device=None):
        self.device = device or Config.DEVICE
        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, data_loader):
        scores, ids = [], []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Predict'):
                ids.extend(batch.pop('id'))
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                scores.extend(torch.sigmoid(self.model(**inputs)).cpu().numpy().reshape(-1).tolist())
        return pd.DataFrame({'id': ids, 'toxic': scores})
