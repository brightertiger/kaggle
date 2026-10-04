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
        self.config.num_classes = len(self.category_mapping)
        self.model = self._load_model()
        self.test_loader = None

    def _load_categories(self) -> List[str]:
        categories_path = os.path.join(self.config.data_path, 'categories.pkl')
        with open(categories_path, 'rb') as f:
            categories = pickle.load(f)
        return [cat.removesuffix('.csv') for cat in categories]

    def _load_model(self) -> torch.nn.Module:
        model = ResNetClassifier(
            model_name=self.model_name,
            num_classes=self.config.num_classes,
            pretrained=False,
        )
        
        state = torch.load(self.model_path, map_location=self.config.device, weights_only=True)
        
        if 'state_dict' in state:
            if state.get('categories', self.category_mapping) != self.category_mapping:
                raise ValueError('Checkpoint category order does not match categories.pkl')
            if state.get('model_name', self.model_name) != self.model_name:
                raise ValueError('Checkpoint architecture does not match model_name')
            self.config.image_size = state.get('image_size', self.config.image_size)
            model_state = state['state_dict']
        else:
            model_state = state
        
        if any(key.startswith('module.') for key in model_state.keys()):
            model_state = {key.removeprefix('module.'): value
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
        if test_df.empty:
            raise ValueError('Test data must not be empty')
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
        
        if not 1 <= top_k <= len(self.category_mapping):
            raise ValueError('top_k must be between 1 and the number of categories')
        predictions_df = self.predict(test_df)
        scores = predictions_df.drop(columns='key_id').to_numpy()
        top_indices = np.argsort(-scores, axis=1, kind='stable')[:, :top_k]
        words = [' '.join(self.category_mapping[idx].replace(' ', '_') for idx in row)
                 for row in top_indices]
        final_submission = pd.DataFrame({'key_id': predictions_df['key_id'], 'word': words})
        
        os.makedirs(os.path.dirname(os.fspath(output_path)) or '.', exist_ok=True)
        final_submission.to_csv(output_path, index=False)
        
        print(f"Submission saved to {output_path}")
        return final_submission
