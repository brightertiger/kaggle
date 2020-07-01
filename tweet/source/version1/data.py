import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer


PARAMS = {}
PARAMS['vocab_file'] = '/root/model/pretrain/vocab.json'
PARAMS['merges_file'] = '/root/model/pretrain/merges.txt'
PARAMS['lowercase'] = True
PARAMS['add_prefix_space'] = True
TOKENIZER = ByteLevelBPETokenizer(**PARAMS)
MAX_LEN = 200

DLOADS = {}
DLOADS['batch_size'] = 4
DLOADS['num_workers'] = 2
DLOADS['drop_last'] = True

class TrainDataset(Dataset):
    
    def __init__(self, subset):
        self.data = pd.read_csv('/root/data/process/train.csv')
        self.data['text'] = self.data['text'].fillna(' ')
        self.data = self.data[self.data['text'] != ' ']
        self.data = self.data[self.data['subset'] != subset]
        self.data = self.data.reset_index(drop=True)
        return None

    def __len__(self):
        return len(self.data)
    
    def __text__(self, idx):
        text = self.data.loc[idx, 'text'].lower()
        sentiment = self.data.loc[idx, 'sentiment'].lower().strip()
        text = " " + " ".join(text.split())
        encodes = TOKENIZER.encode(text)
        sentiment = TOKENIZER.encode(sentiment).ids
        tokens = [0] + sentiment + [2, 2] + encodes.ids + [2]
        offsets = [(0, 0)] * 4 + encodes.offsets + [(0, 0)]
        padding = MAX_LEN - len(tokens)
        if padding > 0:
            tokens += [1] * padding
            offsets += [(0, 0)] * padding
        tokens = torch.tensor(tokens)
        masks = torch.where(tokens != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)
        return tokens, masks, offsets
        
    def __label__(self, idx, offsets):
        text = self.data.loc[idx, 'text'].lower()
        select = self.data.loc[idx, 'selected_text'].lower()
        text = " " + " ".join(text.split())
        select = " " + " ".join(select.split())
        length = len(select) - 1
        start, end = None, None
        for ind in (i for i, e in enumerate(text) if e == select[1]):
            if " " + text[ind: ind + length] == select:
                start = ind
                end = ind + length - 1
                break
        char_targets = [0] * len(text)
        if start != None and end != None:
            for ct in range(start, end + 1):
                char_targets[ct] = 1
        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx
    
    def __getitem__(self, index):
        data = {}        
        tokens, masks, offsets = self.__text__(index)
        start_idx, end_idx = self.__label__(index, offsets)
        aux_label = [0] * 199
        aux_label[start_idx : end_idx] = [1] * (end_idx - start_idx + 1)
        aux_label = torch.tensor(aux_label).reshape(-1,)
        data['tokens'] = tokens
        data['masks'] = masks
        data['start_idx'] = start_idx
        data['end_idx'] = end_idx
        data['aux_label'] = aux_label
        return data
    
class ValidDataset(Dataset):
    
    def __init__(self, subset):
        self.data = pd.read_csv('/root/data/process/train.csv')
        self.data['text'] = self.data['text'].fillna(' ')
        self.data = self.data[self.data['text'] != ' ']
        self.data = self.data[self.data['subset'] == subset]
        self.data = self.data.reset_index(drop=True)
        return None

    def __len__(self):
        return len(self.data)
    
    def __text__(self, idx):
        text = self.data.loc[idx, 'text'].lower()
        sentiment = self.data.loc[idx, 'sentiment'].lower().strip()
        text = " " + " ".join(text.split())
        encodes = TOKENIZER.encode(text)
        sentiment = TOKENIZER.encode(sentiment).ids
        tokens = [0] + sentiment + [2, 2] + encodes.ids + [2]
        offsets = [(0, 0)] * 4 + encodes.offsets + [(0, 0)]
        padding = MAX_LEN - len(tokens)
        if padding > 0:
            tokens += [1] * padding
            offsets += [(0, 0)] * padding
        tokens = torch.tensor(tokens)
        masks = torch.where(tokens != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)
        return tokens, masks, offsets
        
    def __label__(self, idx, offsets):
        text = self.data.loc[idx, 'text'].lower()
        select = self.data.loc[idx, 'selected_text'].lower()
        text = " " + " ".join(text.split())
        select = " " + " ".join(select.split())
        length = len(select) - 1
        start, end = None, None
        for ind in (i for i, e in enumerate(text) if e == select[1]):
            if " " + text[ind: ind + length] == select:
                start = ind
                end = ind + length - 1
                break
        char_targets = [0] * len(text)
        if start != None and end != None:
            for ct in range(start, end + 1):
                char_targets[ct] = 1
        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx
    
    def __getitem__(self, index):
        data = {}        
        tokens, masks, offsets = self.__text__(index)
        start_idx, end_idx = self.__label__(index, offsets)
        aux_label = [0] * 199
        aux_label[start_idx : end_idx] = [1] * (end_idx - start_idx + 1)
        aux_label = torch.tensor(aux_label).reshape(-1,)
        data['tokens'] = tokens
        data['masks'] = masks
        data['start_idx'] = start_idx
        data['end_idx'] = end_idx
        data['aux_label'] = aux_label
        return data
    
def trainLoader(fold):
    train_data = TrainDataset(fold)
    valid_data = ValidDataset(fold)
    train_data = DataLoader(train_data, **DLOADS)
    valid_data = DataLoader(valid_data, **DLOADS)
    return train_data, valid_data
