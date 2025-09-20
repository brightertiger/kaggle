import pandas as pd
import numpy as np
from ..utils.config import Config

class DataPreprocessor:
    def __init__(self):
        self.seed = Config.SEED
    
    def prepare_english_data(self, data_dir):
        columns = ['id', 'source', 'lang', 'comment_text', 'toxic']
        
        # Load datasets
        data1 = pd.read_csv(f'{data_dir}/raw/jigsaw-toxic-comment-train.csv')
        data2 = pd.read_csv(f'{data_dir}/raw/jigsaw-unintended-bias-train.csv')
        data3 = pd.read_csv(f'{data_dir}/raw/extra_english.csv')
        
        # Process labels
        labels = ['toxic', 'severe_toxicity']
        data1['filter'] = 1.
        data2['filter'] = (data2[labels].sum(axis=1) > 0).astype(int)
        data3['filter'] = 1.
        
        # Convert toxic labels
        data2['toxic'] = (data2['toxic'] >= 0.5).astype(int)
        
        # Add metadata
        data1['source'] = '2020-train'
        data2['source'] = '2019-train'
        data1['lang'] = 'en'
        data2['lang'] = 'en'
        
        # Combine datasets
        train_data = pd.concat([data1, data2, data3], ignore_index=True)
        train_data = train_data[train_data['filter'] == 1]
        train_data = train_data[columns].reset_index(drop=True)
        
        # Split data
        train_data = train_data.sample(frac=0.8, random_state=self.seed)
        valid_data = train_data.sample(frac=0.2, random_state=self.seed)
        
        # Save processed data
        train_data.to_csv(f'{data_dir}/process/english/train_english.csv', index=False)
        valid_data.to_csv(f'{data_dir}/process/english/valid_english.csv', index=False)
        
        print(f"English data prepared: {len(train_data)} train, {len(valid_data)} valid")
        return train_data, valid_data
    
    def prepare_foreign_data(self, data_dir):
        # Load foreign datasets
        foreign_data = pd.read_csv(f'{data_dir}/raw/jigsaw-unintended-bias-train.csv')
        
        # Filter non-English data
        foreign_data = foreign_data[foreign_data['lang'] != 'en']
        foreign_data = foreign_data.reset_index(drop=True)
        
        # Process labels
        foreign_data['toxic'] = (foreign_data['toxic'] >= 0.5).astype(int)
        
        # Split data
        train_data = foreign_data.sample(frac=0.8, random_state=self.seed)
        valid_data = foreign_data.sample(frac=0.2, random_state=self.seed)
        
        # Save processed data
        train_data.to_csv(f'{data_dir}/process/foreign/train_foreign.csv', index=False)
        valid_data.to_csv(f'{data_dir}/process/foreign/valid_foreign.csv', index=False)
        
        print(f"Foreign data prepared: {len(train_data)} train, {len(valid_data)} valid")
        return train_data, valid_data
    
    def create_pseudo_labels(self, data_dir):
        # Load adversarial data
        english_adverse = pd.read_csv(f'{data_dir}/process/english/adverse.csv')
        foreign_adverse = pd.read_csv(f'{data_dir}/process/foreign/adverse.csv')
        subtitle_adverse = pd.read_csv(f'{data_dir}/process/subtitle/adverse.csv')
        
        # Create pseudo labels based on adversarial scores
        english_adverse['weight'] = english_adverse['score']
        foreign_adverse['weight'] = foreign_adverse['score']
        subtitle_adverse['weight'] = subtitle_adverse['score']
        
        # Combine datasets
        pseudo_data = pd.concat([english_adverse, foreign_adverse, subtitle_adverse], ignore_index=True)
        
        # Save pseudo data
        pseudo_data.to_csv(f'{data_dir}/process/pseudo/train_combine.csv', index=False)
        
        print(f"Pseudo labels created: {len(pseudo_data)} samples")
        return pseudo_data
    
    def process_all_data(self, data_dir):
        print("Preparing English data...")
        self.prepare_english_data(data_dir)
        
        print("Preparing foreign data...")
        self.prepare_foreign_data(data_dir)
        
        print("Creating pseudo labels...")
        self.create_pseudo_labels(data_dir)
        
        print("Data preprocessing completed!")
