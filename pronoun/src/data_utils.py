"""GAP table validation, mention tokenization, and batch construction."""
import re
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


REQUIRED_COLUMNS = ['ID', 'Text', 'Pronoun', 'Pronoun-offset',
                    'A', 'A-offset', 'B', 'B-offset', 'URL']


def read_gap(path, labeled=False):
    data = pd.read_csv(path, sep='\t')
    required = REQUIRED_COLUMNS + (['A-coref', 'B-coref'] if labeled else [])
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if data.empty or data[required].isna().any().any() or data['ID'].duplicated().any():
        raise ValueError(f'{path}: expected nonempty data, unique IDs, and no missing values')
    for _, row in data.iterrows():
        for mention in ('A', 'B', 'Pronoun'):
            offset = int(row[f'{mention}-offset'])
            if offset != row[f'{mention}-offset'] or offset < 0 or row['Text'][offset:offset + len(row[mention])] != row[mention]:
                raise ValueError(f"{row['ID']}: invalid {mention} character offset")
    if labeled:
        for column in ('A-coref', 'B-coref'):
            values = data[column].astype(str).str.lower().map(
                {'true': True, 'false': False, '1': True, '0': False})
            if values.isna().any():
                raise ValueError(f'{column} must contain boolean coreference labels')
            data[column] = values.astype(bool)
        if (data['A-coref'] & data['B-coref']).any():
            raise ValueError('A and B cannot both be coreferent')
    return data


class PronounDataset(Dataset):
    def __init__(self, text_data, features, labels=None, tokenizer=None, max_length=500):
        self.text_data = text_data
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_data)

    def _tokenize_text(self, text):
        tags, tokens = {}, []
        # Handle markers before WordPiece; they never enter the BERT vocabulary.
        for part in re.split(r'(\[A\]|\[B\]|\[P\])', text):
            if part in ('[A]', '[B]', '[P]'):
                if part in tags:
                    raise ValueError(f'Duplicate mention marker {part}')
                tags[part] = len(tokens)
            else:
                tokens.extend(self.tokenizer.tokenize(part))
        if len(tags) != 3:
            raise ValueError('Text must contain A, B, and P mention markers')
        offsets = [tags[tag] for tag in ('[A]', '[B]', '[P]')]
        budget = self.max_length - 2
        if max(offsets) - min(offsets) >= budget:
            raise ValueError('Mention span exceeds max_length; increase the configured limit')
        # Crop context, never silently truncate a target mention.
        start = max(0, min(min(offsets), max(offsets) - budget + 1))
        tokens = tokens[start:start + budget]
        if any(offset - start >= len(tokens) for offset in offsets):
            raise ValueError('A mention marker must precede a token')
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        return self.tokenizer.convert_tokens_to_ids(tokens), [x - start + 1 for x in offsets]

    def __getitem__(self, idx):
        token_ids, offsets = self._tokenize_text(self.text_data.iloc[idx, 0])
        return {
            'text': token_ids, 'offset': offsets,
            'feature_a': self.features[0].iloc[idx].to_numpy(dtype=np.float32),
            'feature_b': self.features[1].iloc[idx].to_numpy(dtype=np.float32),
            'labels': int(self.labels.iloc[idx, 0]) if self.labels is not None else -1,
        }


def collate_fn(batch):
    tokens = torch.zeros((len(batch), max(len(x['text']) for x in batch)), dtype=torch.long)
    for i, row in enumerate(batch):
        tokens[i, :len(row['text'])] = torch.tensor(row['text'], dtype=torch.long)
    return (tokens,
            torch.tensor([x['offset'] for x in batch], dtype=torch.long),
            torch.from_numpy(np.stack([x['feature_a'] for x in batch])),
            torch.from_numpy(np.stack([x['feature_b'] for x in batch])),
            torch.tensor([x['labels'] for x in batch], dtype=torch.long))


def create_data_loaders(train_dataset, val_dataset, batch_size, num_workers=0):
    options = dict(batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    return (DataLoader(train_dataset, shuffle=True, **options),
            DataLoader(val_dataset, shuffle=False, **options))


def load_and_process_data(train_path: str, val_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return read_gap(train_path, labeled=True), read_gap(val_path, labeled=True)
