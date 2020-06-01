import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class Data(Dataset):
    
    aux_labels = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
    
    def __init__(self, tokenizer, data):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        return None

    def __len__(self):
        return len(self.data)

    def __token__(self, text):
        tokens = []
        for token in self.tokenizer.tokenize(text):
            tokens.append(token)
        tokens = ["[CLS]"] + tokens[:220] + ["[SEP]"]
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens += [0] * (222 - len(tokens))
        tokens = np.array(tokens)
        return tokens

    def __getitem__(self, idx):
        text = self.data.at[idx, 'comment_text']
        text = self.__token__(text)
        weight = self.data.at[idx, 'weight']
        weight = np.array(weight).astype(float)
        labels = np.array(self.data.loc[idx,'target']).astype(float)
        aux = np.array(self.data.loc[idx, self.aux_labels]).astype(float)
        idx = [self.data.at[idx, 'id']]
        return {'idx' : idx, 'text' : text, 'weight' : weight, 'labels' : labels, 'aux' : aux}
    