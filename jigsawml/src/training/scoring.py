import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..models.models import XLMRobertaClassifier
from ..data.data_utils import create_test_loader
from ..utils.config import Config

class ModelScorer:
    def __init__(self, model_path, device=Config.DEVICE):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path):
        model = XLMRobertaClassifier()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        return model
    
    def score_dataset(self, test_path, output_path):
        test_loader = create_test_loader(test_path)
        
        scores = []
        ids = []
        
        tq = tqdm(total=len(test_loader) * 24, disable=False)
        
        with torch.no_grad():
            for batch in test_loader:
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
        
        results.to_csv(output_path, index=False)
        print(f"Scoring completed. Results saved to {output_path}")
        
        return results
    
    def score_all_folds(self, model_dir, version, test_path, output_dir):
        for fold in range(Config.N_FOLDS):
            model_path = f"{model_dir}/version{version}/model_{fold}.pt"
            output_path = f"{output_dir}/version{version}/score_{fold}.csv"
            
            print(f"Scoring fold {fold} for version {version}...")
            self.score_dataset(test_path, output_path)
        
        print(f"All folds scored for version {version}")

class ScoringPipeline:
    def __init__(self, model_dir):
        self.model_dir = model_dir
    
    def score_all_models(self, test_path):
        scorer = ModelScorer(f"{self.model_dir}/version1/model_0.pt")
        
        # Score Version 1 models
        scorer.score_all_folds(self.model_dir, 1, test_path, self.model_dir)
        
        # Score Version 2 models
        scorer.score_all_folds(self.model_dir, 2, test_path, self.model_dir)
        
        print("All models scored successfully!")
