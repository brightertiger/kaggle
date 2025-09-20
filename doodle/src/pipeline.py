import pandas as pd
import numpy as np
import pickle
import os
import glob
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple, List

from .config import Config
from .trainer import ModelTrainer
from .scorer import ModelScorer


class DoodlePipeline:
    def __init__(self, config: Config):
        self.config = config
        self.setup_directories()

    def setup_directories(self):
        directories = [
            self.config.data_path,
            self.config.model_path,
            self.config.score_path,
            self.config.submit_path,
            os.path.join(self.config.data_path, 'train'),
            os.path.join(self.config.data_path, 'valid'),
            os.path.join(self.config.data_path, 'test')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def preprocess_data(self, 
                       source_path: str,
                       train_ratio: float = 0.9,
                       random_state: int = 2017) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        categories = glob.glob(os.path.join(source_path, '*'))
        categories = [os.path.basename(cat) for cat in categories]
        
        categories_path = os.path.join(self.config.data_path, 'categories.pkl')
        with open(categories_path, 'wb') as f:
            pickle.dump(categories, f)
        
        print(f"Found {len(categories)} categories")
        
        full_data = pd.DataFrame()
        
        for i, category in enumerate(categories):
            print(f"Processing category {i+1}/{len(categories)}: {category}")
            
            category_file = os.path.join(source_path, category)
            if os.path.isfile(category_file):
                data = pd.read_csv(category_file)
                data = data[['key_id', 'drawing', 'word']]
                full_data = pd.concat([full_data, data], ignore_index=True)
        
        print(f"Total samples: {len(full_data)}")
        
        split = StratifiedShuffleSplit(
            n_splits=1, 
            random_state=random_state, 
            test_size=1-train_ratio
        )
        
        train_idx, valid_idx = next(split.split(full_data, full_data['word']))
        
        train_data = full_data.iloc[train_idx].reset_index(drop=True)
        valid_data = full_data.iloc[valid_idx].reset_index(drop=True)
        
        train_path = os.path.join(self.config.data_path, 'train', 'train.csv')
        valid_path = os.path.join(self.config.data_path, 'valid', 'valid.csv')
        
        train_data.to_csv(train_path, index=False)
        valid_data.to_csv(valid_path, index=False)
        
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(valid_data)}")
        
        return train_data, valid_data

    def train_model(self, 
                   train_df: pd.DataFrame,
                   valid_df: pd.DataFrame,
                   model_name: str = 'resnet50',
                   learning_rate: float = 0.001) -> dict:
        
        trainer = ModelTrainer(
            config=self.config,
            model_name=model_name,
            learning_rate=learning_rate
        )
        
        results = trainer.train(train_df, valid_df)
        
        print(f"Training completed. Best metric: {results['best_metric']:.4f}")
        return results

    def generate_predictions(self,
                           test_df: pd.DataFrame,
                           model_path: str,
                           model_name: str,
                           output_path: str) -> pd.DataFrame:
        
        scorer = ModelScorer(
            config=self.config,
            model_path=model_path,
            model_name=model_name
        )
        
        submission = scorer.generate_submission(test_df, output_path)
        return submission

    def run_full_pipeline(self,
                         source_data_path: str,
                         test_data_path: str,
                         model_name: str = 'resnet50',
                         learning_rate: float = 0.001) -> dict:
        
        print("Starting full pipeline...")
        
        train_df, valid_df = self.preprocess_data(source_data_path)
        
        train_results = self.train_model(train_df, valid_df, model_name, learning_rate)
        
        test_df = pd.read_csv(test_data_path)
        model_path = os.path.join(
            self.config.model_path, 
            model_name, 
            f'{model_name}_best.pth'
        )
        
        output_path = os.path.join(
            self.config.submit_path, 
            f'{model_name}_submission.csv'
        )
        
        submission = self.generate_predictions(
            test_df, model_path, model_name, output_path
        )
        
        return {
            'training_results': train_results,
            'submission_path': output_path,
            'model_path': model_path
        }
