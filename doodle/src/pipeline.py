import pandas as pd
import pickle
import os
import glob
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple

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
                       train_ratio: float = None,
                       random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

        train_ratio = self.config.train_ratio if train_ratio is None else train_ratio
        random_state = self.config.random_seed if random_state is None else random_state
        if not 0 < train_ratio < 1:
            raise ValueError('train_ratio must be between 0 and 1')
        category_files = sorted(glob.glob(os.path.join(source_path, '*.csv')))
        if not category_files:
            raise ValueError(f'No category CSV files found in {source_path}')
        frames = []
        for category_file in category_files:
            print(f"Processing {os.path.basename(category_file)}")
            data = pd.read_csv(
                category_file, usecols=['key_id', 'drawing', 'word'],
                dtype={'key_id': str, 'word': str},
                nrows=self.config.max_samples_per_class,
            )
            if data.empty or data.isna().any().any():
                raise ValueError(f'Empty or missing required values in {category_file}')
            frames.append(data)
        full_data = pd.concat(frames, ignore_index=True)
        categories = sorted(full_data['word'].unique().tolist())
        print(f"Found {len(categories)} categories")
        
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
        with open(os.path.join(self.config.data_path, 'categories.pkl'), 'wb') as f:
            pickle.dump(categories, f)
        self.config.num_classes = len(categories)
        
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(valid_data)}")
        
        return train_data, valid_data

    def train_model(self, 
                   train_df: pd.DataFrame,
                   valid_df: pd.DataFrame,
                   model_name: str = 'resnet50',
                   learning_rate: float = None) -> dict:
        
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
                         learning_rate: float = None) -> dict:
        
        print("Starting full pipeline...")
        
        train_df, valid_df = self.preprocess_data(source_data_path)
        
        train_results = self.train_model(train_df, valid_df, model_name, learning_rate)
        
        test_df = pd.read_csv(test_data_path, dtype={'key_id': str})
        model_path = os.path.join(
            self.config.model_path, 
            model_name, 
            f'{model_name}_best.pth'
        )
        
        output_path = os.path.join(
            self.config.submit_path, 
            f'{model_name}_submission.csv'
        )
        
        self.generate_predictions(
            test_df, model_path, model_name, output_path
        )
        
        return {
            'training_results': train_results,
            'submission_path': output_path,
            'model_path': model_path
        }
