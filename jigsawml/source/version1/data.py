import re
import string
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer
warnings.filterwarnings("ignore")

PARAMS = {}
PARAMS['max_length'] = 300
PARAMS['pad_to_max_length'] = True
PARAMS['return_attention_mask'] = True
PARAMS['truncation_strategy'] = 'longest_first'
PARAMS['add_special_tokens'] = True
PARAMS['do_lower_case'] = False
TOKENIZER = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large', do_lower_case=False)

DLOADS = {}
DLOADS['batch_size'] = 8
DLOADS['num_workers'] = 10
DLOADS['drop_last'] = True

FEATS = ['use_' + str(x) for x in range(512)]

def tokenizeText(text):
    text = str(text) + ' '
    if len(text.split()) >= 200:
        text = text.split()
        text = text[:200] + text[-50:]
        text = ' '.join(text)
    text = TOKENIZER.encode_plus(text, **PARAMS)
    tokens = text['input_ids']
    attens = text['attention_mask']
    return tokens, attens


class TrainDataset(Dataset):
    
    def __init__(self, subset):
        self.source = pd.read_csv('/root/jigsaw/data/process/pseudo/train_combine.csv')
        self.source = self.source[self.source['source'] == '2020-train']
        self.source = self.source.sample(frac=1., random_state=2017)
        self.source = self.source.reset_index(drop=True)
        self.source['fold'] = self.source.index % 5
        self.source = self.source[self.source['fold'] == subset]
        self.source = self.source.reset_index(drop=True)
        print('Data:', self.source.shape)
        self.seed = 1
        return None                          
                          
    def __len__(self):
        return 50000
    
    def epoch(self, epoch):
        self.data = self.source.sample(frac=1., random_state=2017)
        self.data = self.data.reset_index(drop=True)
        return None
        
    def text(self, idx):
        text = str(self.data.loc[idx, 'comment_text']) + ' '
        tokens, attens = tokenizeText(text)
        tokens = np.array(tokens).astype(int)
        attens = np.array(attens).astype(int)
        return tokens, attens
    
    def label(self, idx):
        label = self.data.loc[idx, 'toxic']
        noise = np.random.uniform(low=0.0, high=0.1)
        if label > 0.5:
            label = label - noise
        else:
            label = label + noise
        return np.array(label).astype(float)
    
    def weight(self, idx):
        weight = self.data.loc[idx, 'weight']
        return np.array(weight).astype(float)
       
    def __getitem__(self, idx):
        data = {}
        data['label'] = self.label(idx)
        data['weight'] = self.weight(idx)
        data['tokens'], data['attens'] = self.text(idx)
        return data

class ValidDataset(Dataset):
    
    def __init__(self):
        self.data = pd.read_csv('../../data/process/foreign/valid_foreign.csv')
        self.data = self.data[self.data['original'] == 1]
        self.data = self.data.reset_index(drop=True)
        return None
    
    def __len__(self):
        return len(self.data)
    
    def label(self, idx):
        label = self.data.loc[idx, 'toxic']
        return np.array(label).astype(float)
    
    def text(self, idx):
        text = str(self.data.loc[idx, 'comment_text']) + ' '
        tokens, attens = tokenizeText(text)
        tokens = np.array(tokens).astype(int)
        attens = np.array(attens).astype(int)
        return tokens, attens
        
    def __getitem__(self, idx):
        data = {}
        data['label'] = self.label(idx)
        data['weight'] = 1.
        data['tokens'], data['attens'] = self.text(idx)
        return data
    
def trainLoader(subset):
    train_data = TrainDataset(subset)
    valid_data = ValidDataset()
    train_data = DataLoader(train_data, **DLOADS)
    valid_data = DataLoader(valid_data, **DLOADS)
    return train_data, valid_data
