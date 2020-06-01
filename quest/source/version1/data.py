import re
import string
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

def cleanText(text):
    text = text.split()
    clean = []
    for word in text:
        if word in VOCAB:
            clean.append(word)
        elif re.sub(r'[^\w\s]','',word) in VOCAB:
            clean.append(re.sub(r'[^\w\s]','',word))
        else:
            word = re.sub(r'[^\w\s]',' ',word)
            words = word.split()
            for sub in words:
                if sub in VOCAB:
                    clean.append(sub)
    return ' '.join(clean)

def tokenizeText(text):
    global TOKENIZER
    text = text.replace('\n'," ").lower()
    backup = text
    text = TOKENIZER.tokenize(text)
    if len(text) >= 400:
        text = backup
        text = cleanText(text)
        text = TOKENIZER.tokenize(text)
    text = ['[CLS]'] + text[:500] + ['[SEP]']
    tokens = TOKENIZER.convert_tokens_to_ids(text)
    return tokens
  
class ModelDataset(Dataset):
    
    def __init__(self, mode, fold):
        prefix = '../../data/split/'
        self.label = pd.read_csv(prefix + 'label_{}_{}.csv'.format(mode, fold))
        self.data = pd.read_csv(prefix + 'data_{}_{}.csv'.format(mode, fold))
        self.text =  pd.read_csv(prefix + 'text_{}_{}.csv'.format(mode, fold))
        self.index = self.label['qa_id'].tolist()
        return None
    
    def __len__(self):
        return len(self.data)
    
    def question(self, idx):
        title = self.text[self.text['qa_id'] == idx].iat[0,1]
        question = self.text[self.text['qa_id'] == idx].iat[0,2]
        question = title + ' [SEP] ' + question
        tokens = tokenizeText(question)
        tokens = tokens + [0] * (512 - len(tokens))
        return np.array(tokens).astype(int)
    
    def answer(self, idx):
        answer = self.text[self.text['qa_id'] == idx].iat[0,3]
        tokens = tokenizeText(answer)
        tokens = tokens + [0] * (512 - len(tokens))
        return np.array(tokens).astype(int)
    
    def labels(self, idx):
        labels = self.label[self.label['qa_id'] == idx].iloc[:,1:]
        return np.array(labels)
        
    def __getitem__(self, idx):
        idx = self.index[idx]
        data = {}
        data['label'] = self.labels(idx)
        data['question'] = self.question(idx)
        data['answer'] = self.answer(idx)
        return data
    
def dataLoader(fold, batch_size):
    train_data = ModelDataset('train', fold)
    valid_data = ModelDataset('valid', fold)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_data = DataLoader(valid_data, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_data, valid_data
    