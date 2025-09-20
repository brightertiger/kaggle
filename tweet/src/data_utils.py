import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from typing import Tuple, Dict, Any
from .config import Config

class TweetDataset(Dataset):
    
    def __init__(self, data_path: str, subset: int, config: Config, is_training: bool = True):
        self.data = pd.read_csv(data_path)
        self.data['text'] = self.data['text'].fillna(' ')
        self.data = self.data[self.data['text'] != ' ']
        
        if is_training:
            if subset >= 0:
                self.data = self.data[self.data['subset'] != subset]
            else:
                self.data = self.data[self.data['subset'] == abs(subset) - 1]
        
        self.data = self.data.reset_index(drop=True)
        self.config = config
        self.tokenizer = self._load_tokenizer()
        
    def _load_tokenizer(self) -> ByteLevelBPETokenizer:
        params = {
            'vocab_file': self.config.data.vocab_file,
            'merges_file': self.config.data.merges_file,
            'lowercase': True,
            'add_prefix_space': True
        }
        return ByteLevelBPETokenizer(**params)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _encode_text(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text = self.data.loc[idx, 'text'].lower()
        sentiment = self.data.loc[idx, 'sentiment'].lower().strip()
        text = " " + " ".join(text.split())
        
        encodes = self.tokenizer.encode(text)
        sentiment_ids = self.tokenizer.encode(sentiment).ids
        
        tokens = [0] + sentiment_ids + [2, 2] + encodes.ids + [2]
        offsets = [(0, 0)] * 4 + encodes.offsets + [(0, 0)]
        
        padding = self.config.data.max_length - len(tokens)
        if padding > 0:
            tokens += [1] * padding
            offsets += [(0, 0)] * padding
            
        tokens = torch.tensor(tokens)
        masks = torch.where(tokens != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)
        
        return tokens, masks, offsets
    
    def _compute_labels(self, idx: int, offsets: torch.Tensor) -> Tuple[int, int]:
        text = self.data.loc[idx, 'text'].lower()
        selected_text = self.data.loc[idx, 'selected_text'].lower()
        
        text = " " + " ".join(text.split())
        selected_text = " " + " ".join(selected_text.split())
        
        length = len(selected_text) - 1
        start, end = None, None
        
        for ind in (i for i, e in enumerate(text) if e == selected_text[1]):
            if " " + text[ind: ind + length] == selected_text:
                start = ind
                end = ind + length - 1
                break
        
        char_targets = [0] * len(text)
        if start is not None and end is not None:
            for ct in range(start, end + 1):
                char_targets[ct] = 1
        
        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        
        if target_idx:
            start_idx = target_idx[0]
            end_idx = target_idx[-1]
        else:
            start_idx = end_idx = 0
            
        return start_idx, end_idx
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        tokens, masks, offsets = self._encode_text(index)
        start_idx, end_idx = self._compute_labels(index, offsets)
        
        aux_label = [0] * (self.config.data.max_length - 1)
        aux_label[start_idx: end_idx + 1] = [1] * (end_idx - start_idx + 1)
        aux_label = torch.tensor(aux_label).reshape(-1,)
        
        return {
            'tokens': tokens,
            'masks': masks,
            'start_idx': torch.tensor(start_idx),
            'end_idx': torch.tensor(end_idx),
            'aux_label': aux_label
        }

def create_data_loaders(config: Config, fold: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TweetDataset(
        config.data.train_path, 
        subset=fold, 
        config=config, 
        is_training=True
    )
    
    valid_dataset = TweetDataset(
        config.data.train_path, 
        subset=-fold-1, 
        config=config, 
        is_training=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        drop_last=True,
        shuffle=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        drop_last=False,
        shuffle=False
    )
    
    return train_loader, valid_loader

def create_submission_dataset(config: Config, test_path: str) -> DataLoader:
    test_data = pd.read_csv(test_path)
    test_data['text'] = test_data['text'].fillna(' ')
    test_data = test_data[test_data['text'] != ' ']
    test_data = test_data.reset_index(drop=True)
    
    dataset = TweetDataset(
        test_path,
        subset=0,
        config=config,
        is_training=False
    )
    
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        drop_last=False,
        shuffle=False
    )
