import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any
import re
from pytorch_pretrained_bert import BertTokenizer

class PronounDataset(Dataset):
    def __init__(self, 
                 text_data: pd.DataFrame, 
                 features: Tuple[pd.DataFrame, pd.DataFrame], 
                 labels: pd.DataFrame = None,
                 tokenizer: BertTokenizer = None):
        self.text_data = text_data
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.text_data)
    
    def _tokenize_text(self, text: str) -> Tuple[List[int], List[int]]:
        tags = {}
        tokens = []
        
        for token in self.tokenizer.tokenize(text):
            if token in ("[A]", "[B]", "[P]"):
                tags[token] = len(tokens)
                continue
            tokens.append(token)
        
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        offsets = [tags["[A]"] + 1, tags["[B]"] + 1, tags["[P]"] + 1]
        
        return token_ids, offsets
    
    def __getitem__(self, idx):
        text = self.text_data.iloc[idx, 0]
        token_ids, offsets = self._tokenize_text(text)
        
        feature_a = np.array(self.features[0].iloc[idx, :])
        feature_b = np.array(self.features[1].iloc[idx, :])
        
        if self.labels is not None:
            labels = np.array(self.labels.iloc[idx, :])
        else:
            labels = np.array([0, 0, 0])
        
        return {
            'text': token_ids,
            'offset': offsets,
            'feature_a': feature_a,
            'feature_b': feature_b,
            'labels': labels
        }

def collate_fn(batch):
    text = [x['text'] for x in batch]
    labels = [x['labels'] for x in batch]
    feature_a = [x['feature_a'] for x in batch]
    feature_b = [x['feature_b'] for x in batch]
    offset = [x['offset'] for x in batch]
    
    max_length = min(500, max(len(x) for x in text))
    
    tokens = np.zeros((len(text), max_length))
    for idx, data in enumerate(text):
        tokens[idx, :len(data)] = np.array(data[:max_length])
    
    tokens = torch.from_numpy(tokens)
    offset = torch.stack([torch.LongTensor(x) for x in offset], dim=0)
    feature_a = torch.stack([torch.from_numpy(x).float() for x in feature_a], dim=0)
    feature_b = torch.stack([torch.from_numpy(x).float() for x in feature_b], dim=0)
    labels = torch.stack([torch.from_numpy(x) for x in labels], dim=0)
    
    return tokens, offset, feature_a, feature_b, labels

def create_data_loaders(train_dataset, val_dataset, batch_size: int, num_workers: int = 1):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader

def load_and_process_data(train_path: str, val_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_csv(train_path, sep='\t')
    val_data = pd.read_csv(val_path, sep='\t')
    
    combined_data = pd.concat([train_data, val_data], ignore_index=True)
    
    combined_data['fold'] = combined_data.index.map(lambda x: (x % 5) + 1)
    
    return train_data, val_data
