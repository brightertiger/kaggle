import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re
import pickle

class Data(Dataset):

    def __init__(self, tokenizer, data, labels=None):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.labels = labels
        return None

    def __len__(self):
        return len(self.data)

    def __token__(self, text):
        tokens = []
        tokens = self.tokenizer.encode(text)
        tokens = tokens[:220]
        tokens += [0] * (220 - len(tokens))
        tokens = np.array(tokens)
        return tokens

    def __getitem__(self, idx):
        text = self.data.at[idx, 'comment_text']
        text = self.__token__(text)
        weight = np.array(self.data.at[idx, 'weight']).astype(float)
        if self.labels is not None:
            labels = np.array(self.data.loc[idx, 'target']).astype(float)
        else:
            labels = np.array([0]).astype(int)
        idx = [self.data.at[idx, 'id']]
        return {'idx' : idx, 'text' : text, 'weight' : weight, 'labels' : labels}