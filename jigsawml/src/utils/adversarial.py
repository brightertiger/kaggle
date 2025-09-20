import pandas as pd
import numpy as np
import lightgbm as lgb
from .config import Config

class AdversarialGenerator:
    def __init__(self):
        self.lgb_params = Config.LGB_PARAMS
        self.use_features = Config.USE_FEATURES
    
    def create_adversarial_data(self, train_path, valid_path, test_path, output_path):
        train = pd.read_csv(train_path)
        valid = pd.read_csv(valid_path)
        test = pd.read_csv(test_path)
        
        data = pd.concat([train, valid, test], ignore_index=True)
        labels = [0] * len(train) + [1] * (len(valid) + len(test))
        data['label'] = labels
        data = data.sample(frac=1., random_state=Config.SEED).reset_index(drop=True)
        
        cv_results = []
        
        for fold in range(Config.N_FOLDS):
            train_data = data[data.index % Config.N_FOLDS != fold].reset_index(drop=True)
            valid_data = data[data.index % Config.N_FOLDS == fold].reset_index(drop=True)
            
            train_matrix = lgb.Dataset(
                train_data[self.use_features], 
                np.array(train_data['label']).reshape(-1,)
            )
            valid_matrix = lgb.Dataset(
                valid_data[self.use_features], 
                np.array(valid_data['label']).reshape(-1,)
            )
            
            model = lgb.train(
                params=self.lgb_params,
                train_set=train_matrix,
                valid_sets=[train_matrix, valid_matrix],
                num_boost_round=250,
                early_stopping_rounds=50,
                verbose_eval=50
            )
            
            valid_scores = model.predict(valid_data[self.use_features])
            
            fold_result = valid_data[['id', 'source', 'lang']].copy()
            fold_result['score'] = valid_scores
            cv_results.append(fold_result)
        
        combined_results = pd.concat(cv_results, ignore_index=True)
        combined_results.to_csv(output_path, index=False)
        
        return combined_results
    
    def generate_all_adversarial_data(self, data_dir):
        datasets = [
            {
                'name': 'English',
                'train': f'{data_dir}/english/train_english_embed.csv',
                'valid': f'{data_dir}/english/valid_english_embed.csv',
                'test': f'{data_dir}/english/test_english_embed.csv',
                'output': f'{data_dir}/english/adverse.csv'
            },
            {
                'name': 'Subtitle',
                'train': f'{data_dir}/subtitle/subtitle_embed.csv',
                'valid': f'{data_dir}/foreign/valid_foreign_embed.csv',
                'test': f'{data_dir}/foreign/test_foreign_embed.csv',
                'output': f'{data_dir}/subtitle/adverse.csv'
            },
            {
                'name': 'Translation',
                'train': f'{data_dir}/foreign/train_foreign_embed.csv',
                'valid': f'{data_dir}/foreign/valid_foreign_embed.csv',
                'test': f'{data_dir}/foreign/test_foreign_embed.csv',
                'output': f'{data_dir}/foreign/adverse.csv'
            }
        ]
        
        for dataset in datasets:
            print(f"Generating adversarial data for {dataset['name']}...")
            self.create_adversarial_data(
                dataset['train'],
                dataset['valid'],
                dataset['test'],
                dataset['output']
            )
        
        print("All adversarial data generated successfully!")
