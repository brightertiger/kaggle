import re
import string
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer
from ..utils.config import Config

class TextTokenizer:
    def __init__(self):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            Config.MODEL_NAME, 
            do_lower_case=False
        )
        self.params = {
            'max_length': Config.MAX_LENGTH,
            'pad_to_max_length': True,
            'return_attention_mask': True,
            'truncation_strategy': 'longest_first',
            'add_special_tokens': True,
            'do_lower_case': False
        }
    
    def tokenize_text(self, text):
        text = str(text) + ' '
        if len(text.split()) >= 200:
            text = text.split()
            text = text[:200] + text[-50:]
            text = ' '.join(text)
        
        encoded = self.tokenizer.encode_plus(text, **self.params)
        tokens = np.array(encoded['input_ids']).astype(int)
        attention_mask = np.array(encoded['attention_mask']).astype(int)
        return tokens, attention_mask

class TrainDataset(Dataset):
    def __init__(self, subset):
        self.source = pd.read_csv(f'{Config.DATA_DIR}/pseudo/train_combine.csv')
        self.source = self.source[self.source['source'] == '2020-train']
        self.source = self.source.sample(frac=1., random_state=Config.SEED)
        self.source = self.source.reset_index(drop=True)
        self.source['fold'] = self.source.index % Config.N_FOLDS
        self.source = self.source[self.source['fold'] == subset]
        self.source = self.source.reset_index(drop=True)
        print(f'Data: {self.source.shape}')
        
        self.tokenizer = TextTokenizer()
    
    def __len__(self):
        return 50000
    
    def epoch(self, epoch):
        self.data = self.source.sample(frac=1., random_state=Config.SEED)
        self.data = self.data.reset_index(drop=True)
    
    def __getitem__(self, idx):
        text = str(self.data.loc[idx, 'comment_text']) + ' '
        tokens, attention_mask = self.tokenizer.tokenize_text(text)
        
        label = self.data.loc[idx, 'toxic']
        noise = np.random.uniform(low=0.0, high=0.1)
        if label > 0.5:
            label = label - noise
        else:
            label = label + noise
        
        weight = self.data.loc[idx, 'weight']
        
        return {
            'tokens': tokens,
            'attention_mask': attention_mask,
            'label': np.array(label).astype(float),
            'weight': np.array(weight).astype(float)
        }

class ValidDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv(f'{Config.DATA_DIR}/foreign/valid_foreign.csv')
        self.data = self.data[self.data['original'] == 1]
        self.data = self.data.reset_index(drop=True)
        self.tokenizer = TextTokenizer()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.loc[idx, 'comment_text']) + ' '
        tokens, attention_mask = self.tokenizer.tokenize_text(text)
        
        label = self.data.loc[idx, 'toxic']
        
        return {
            'tokens': tokens,
            'attention_mask': attention_mask,
            'label': np.array(label).astype(float),
            'weight': np.array(1.0).astype(float)
        }

class TestDataset(Dataset):
    def __init__(self, test_path):
        self.data = pd.read_csv(test_path)
        self.tokenizer = TextTokenizer()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.loc[idx, 'comment_text']) + ' '
        tokens, attention_mask = self.tokenizer.tokenize_text(text)
        
        sample_id = self.data.loc[idx, 'id']
        
        return {
            'tokens': tokens,
            'attention_mask': attention_mask,
            'id': np.array(sample_id).astype(int)
        }

def create_data_loaders(subset):
    train_dataset = TrainDataset(subset)
    valid_dataset = ValidDataset()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        drop_last=True
    )
    
    return train_loader, valid_loader

def create_test_loader(test_path):
    test_dataset = TestDataset(test_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=24,
        num_workers=Config.NUM_WORKERS,
        drop_last=False
    )
    return test_loader
