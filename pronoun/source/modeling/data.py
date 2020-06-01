import torch
import numpy as np
from torch.utils.data import Dataset

class Data(Dataset):

    def __init__(self, tokenizer, text, features, labels=None):
        super().__init__()
        self.text = text
        self.tokenizer = tokenizer
        self.labels = labels
        self.features = features
        return None

    def __len__(self):
        return len(self.text)

    def __token__(self, text):
        tags = {}
        tokens = []
        for token in self.tokenizer.tokenize(text):
            if token in ("[A]", "[B]", "[P]"):
                tags[token] = len(tokens)
                continue
            tokens.append(token)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        offset = [tags["[A]"] + 1, tags["[B]"] + 1, tags["[P]"] + 1]
        return tokens, offset

    def __getitem__(self, idx):
        text = self.text.iloc[idx,0]
        text, offset = self.__token__(text)
        feature_a = np.array(self.features[0].iloc[idx,:])
        feature_b = np.array(self.features[1].iloc[idx,:])
        if self.labels is not None:
            labels = np.array(self.labels.iloc[idx, :])
        else:
            labels = np.array([0,0,0])
        return {'text' : text, 'offset' : offset, 'feature_a': feature_a, 'feature_b' : feature_b, 'labels' : labels}

def batchWrapper(batch):
    text = list(x['text'] for x in batch)
    labels = list(x['labels'] for x in batch)
    feature_a = list(x['feature_a'] for x in batch)
    feature_b = list(x['feature_b'] for x in batch)
    offset = list(x['offset'] for x in batch)
    length = max(len(x) for x in text)
    length = min(500, length)
    tokens = np.zeros((len(text),length))
    for idx, data in enumerate(text):
        tokens[idx, :len(data)] = np.array(data[:length])
    tokens = torch.from_numpy(tokens)
    offset = torch.stack([torch.LongTensor(x) for x in offset], dim=0)
    feature_a = torch.stack([torch.from_numpy(x).float() for x in feature_a], dim=0)
    feature_b = torch.stack([torch.from_numpy(x).float() for x in feature_b], dim=0)
    labels = torch.stack([torch.from_numpy(x) for x in labels], dim=0)
    return tokens, offset, feature_a, feature_b, labels