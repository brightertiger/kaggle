import torch
import numpy as np
import pandas as pd
import pickle
import os
from typing import List

from .models import ResNetClassifier
from .data_utils import create_test_dataloader
from .config import Config


class ModelScorer:
    def __init__(self, 
                 config: Config,
                 model_path: str,
                 model_name: str = 'resnet50'):
        
        self.config = config
        self.model_path = model_path
        self.model_name = model_name
        
        self.category_mapping = self._load_categories()
        self.model = self._load_model()
        self.test_loader = None

    def _load_categories(self) -> List[str]:
        categories_path = os.path.join(self.config.data_path, 'categories.pkl')
        with open(categories_path, 'rb') as f:
            categories = pickle.load(f)
        return [cat.replace('.csv', '') for cat in categories]

    def _load_model(self) -> torch.nn.Module:
        model = ResNetClassifier(
            model_name=self.model_name,
            num_classes=self.config.num_classes
        )
        
        state = torch.load(self.model_path, map_location=self.config.device)
        
        if 'state_dict' in state:
            model_state = state['state_dict']
        else:
            model_state = state
        
        if any(key.startswith('module.') for key in model_state.keys()):
            model_state = {key.replace('module.', ''): value 
                          for key, value in model_state.items()}
        
        model.load_state_dict(model_state)
        model.to(self.config.device)
        model.eval()
        
        print(f"Model loaded from {self.model_path}")
        return model

    def prepare_test_data(self, test_df: pd.DataFrame):
        self.test_loader = create_test_dataloader(
            test_df, self.category_mapping, self.config
        )

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        self.prepare_test_data(test_df)
        
        all_predictions = []
        key_ids = []
        
        with torch.no_grad():
            for sample in self.test_loader:
                images = sample['image'].to(self.config.device)
                batch_key_ids = sample['key_id']
                
                outputs = self.model(images)
                predictions = torch.softmax(outputs, dim=1).cpu().numpy()
                
                all_predictions.append(predictions)
                key_ids.extend(batch_key_ids)
        
        all_predictions = np.vstack(all_predictions)
        
        result_df = pd.DataFrame(all_predictions)
        result_df['key_id'] = key_ids
        
        return result_df

    def generate_submission(self, 
                          test_df: pd.DataFrame,
                          output_path: str,
                          top_k: int = 3) -> pd.DataFrame:
        
        predictions_df = self.predict(test_df)
        
        def get_top_k_categories(row):
            scores = row.drop('key_id').values
            top_indices = np.argsort(scores)[-top_k:][::-1]
            top_categories = [self.category_mapping[idx] for idx in top_indices]
            return ' '.join(top_categories)
        
        submission_df = predictions_df.copy()
        submission_df['word'] = submission_df.apply(get_top_k_categories, axis=1)
        
        final_submission = submission_df[['key_id', 'word']].copy()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_submission.to_csv(output_path, index=False)
        
        print(f"Submission saved to {output_path}")
        return final_submission
