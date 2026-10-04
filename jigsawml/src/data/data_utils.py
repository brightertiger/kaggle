import hashlib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer
from ..utils.config import Config


class TextTokenizer:
    def __init__(self, config=None):
        self.config = config or Config()
        self.tokenizer = None if self.config.TINY else XLMRobertaTokenizer.from_pretrained(self.config.MODEL_NAME)

    def tokenize_text(self, text):
        words = str(text).split()
        if len(words) >= 200:
            words = words[:200] + words[-50:]
        text = ' '.join(words)
        if self.config.TINY:
            # Stable offline token IDs; only used to smoke-test the same model path.
            ids = [0] + [4 + int(hashlib.sha256(w.encode()).hexdigest(), 16) %
                         (self.config.VOCAB_SIZE - 4) for w in words[:self.config.MAX_LENGTH - 2]] + [2]
            mask = [1] * len(ids)
            padding = self.config.MAX_LENGTH - len(ids)
            return np.array(ids + [1] * padding), np.array(mask + [0] * padding)
        encoded = self.tokenizer(text, max_length=self.config.MAX_LENGTH,
                                 padding='max_length', truncation=True, return_attention_mask=True)
        return np.array(encoded['input_ids']), np.array(encoded['attention_mask'])


class TrainDataset(Dataset):
    def __init__(self, subset, config=None):
        self.config = config or Config()
        if not 0 <= subset < self.config.N_FOLDS:
            raise ValueError('subset must be in [0, N_FOLDS)')
        data = pd.read_csv(f'{self.config.DATA_DIR}/pseudo/train_combine.csv')
        # Preserve the original disjoint English training subsets with a fixed
        # multilingual holdout. These are ensemble partitions, not K-fold CV.
        data = data.loc[data['source'] == '2020-train']
        data = data.sample(frac=1, random_state=self.config.SEED).reset_index(drop=True)
        self.source = data.loc[data.index % self.config.N_FOLDS == subset].reset_index(drop=True)
        if self.source.empty:
            raise ValueError(f'No training rows for subset {subset}')
        self.tokenizer = TextTokenizer(self.config)
        self.epoch(0)

    def __len__(self):
        return len(self.source)

    def epoch(self, epoch):
        self.data = self.source.sample(frac=1, random_state=self.config.SEED + epoch).reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens, mask = self.tokenizer.tokenize_text(row['comment_text'])
        label = float(row['toxic'])
        noise = np.random.uniform(0, 0.1)
        label = label - noise if label > 0.5 else label + noise
        return dict(tokens=tokens, attention_mask=mask,
                    label=np.float32(label), weight=np.float32(row['weight']))


class ValidDataset(Dataset):
    def __init__(self, config=None):
        self.config = config or Config()
        self.data = pd.read_csv(f'{self.config.DATA_DIR}/foreign/valid_foreign.csv')
        if 'original' in self.data:
            self.data = self.data.loc[self.data['original'] == 1]
        self.data = self.data.reset_index(drop=True)
        self.tokenizer = TextTokenizer(self.config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens, mask = self.tokenizer.tokenize_text(row['comment_text'])
        return dict(tokens=tokens, attention_mask=mask, label=np.float32(row['toxic']))


class TestDataset(Dataset):
    def __init__(self, test_path, config=None):
        self.data = pd.read_csv(test_path, dtype={'id': str}).rename(columns={'content': 'comment_text'})
        self.tokenizer = TextTokenizer(config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens, mask = self.tokenizer.tokenize_text(row['comment_text'])
        return dict(tokens=tokens, attention_mask=mask, id=str(row['id']))


def create_data_loaders(subset, config=None):
    config = config or Config()
    params = dict(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, drop_last=False)
    return (DataLoader(TrainDataset(subset, config), **params),
            DataLoader(ValidDataset(config), **params))


def create_test_loader(test_path, config=None):
    config = config or Config()
    return DataLoader(TestDataset(test_path, config), batch_size=config.BATCH_SIZE,
                      num_workers=config.NUM_WORKERS, drop_last=False)
