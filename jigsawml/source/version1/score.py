import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
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
DLOADS['batch_size'] = 24
DLOADS['num_workers'] = 10
DLOADS['drop_last'] = False

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

class TestDataset(Dataset):
    
    def __init__(self):
        self.data = pd.read_csv('/root/jigsaw/data/process/foreign/test_english.csv')
        return None
    
    def __len__(self):
        return len(self.data)
    
    def label(self, idx):
        label = self.data.loc[idx, 'id']
        return np.array(label).astype(int)
    
    def text(self, idx):
        text = str(self.data.loc[idx, 'comment_text']) + ' '
        tokens, attens = tokenizeText(text)
        tokens = np.array(tokens).astype(int)
        attens = np.array(attens).astype(int)
        return tokens, attens

    def __getitem__(self, idx):
        data = {}
        data['label'] = self.label(idx)
        data['tokens'], data['attens'] = self.text(idx)
        return data  

def scoreModel(model, data, save):
    model.eval()
    scores = []
    labels = []
    tq = tqdm(total=len(data) * DLOADS['batch_size'], disable=False)
    with torch.no_grad():
        for sample in data:
            label = sample.pop('label')
            for key, value in sample.items():
                sample[key] = value.cuda() 
            preds = torch.sigmoid(model(**sample))
            scores.append(preds.cpu().data.numpy().reshape(-1,1))
            labels.append(label.data.numpy().reshape(-1,1))
            tq.update(BS)
        del sample, label, preds
    tq.close()
    scores = np.vstack(scores)
    labels = np.vstack(labels)
    scores = pd.DataFrame(scores)
    labels = pd.DataFrame(labels)
    scores.columns = ['toxic']
    labels.columns = ['id']
    scores = labels.join(scores)
    print(scores.shape)
    scores.to_csv(save, index=False)
    return None