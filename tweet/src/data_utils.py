"""Shared BPE encoding and character alignment for training and inference."""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from .config import Config


def normalize_text(text: str) -> str:
    return ' ' + ' '.join(str(text).lower().split())


def load_tokenizer(config: Config) -> ByteLevelBPETokenizer:
    tokenizer = ByteLevelBPETokenizer(
        vocab=config.data.vocab_file, merges=config.data.merges_file,
        lowercase=True, add_prefix_space=True,
    )
    for token, expected in [('<s>', 0), ('<pad>', 1), ('</s>', 2)]:
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f'Tokenizer must use RoBERTa special IDs: {token}={expected}')
    return tokenizer


def read_data(path: str, labeled: bool) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={'textID': str}, keep_default_na=False)
    required = {'textID', 'text', 'sentiment'}
    if labeled:
        required.add('selected_text')
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    data['text'] = data['text'].fillna('').astype(str)
    data['sentiment'] = data['sentiment'].str.lower().str.strip()
    if not data['sentiment'].isin(['positive', 'negative', 'neutral']).all():
        raise ValueError('Sentiment must be positive, negative, or neutral')
    if labeled:
        data['selected_text'] = data['selected_text'].fillna('').astype(str)
        data = data[data['text'].str.strip().ne('') & data['selected_text'].str.strip().ne('')]
    return data.reset_index(drop=True)


def encode_text(tokenizer, text: str, sentiment: str, max_length: int):
    encoded = tokenizer.encode(normalize_text(text))
    sentiment_ids = tokenizer.encode(sentiment.lower().strip()).ids
    prefix = [0] + sentiment_ids + [2, 2]
    available = max_length - len(prefix) - 1
    if available < 1:
        raise ValueError('max_length must leave room for tweet tokens after sentiment')
    tweet_ids = encoded.ids[:available]
    tokens = prefix + tweet_ids + [2]
    offsets = [(0, 0)] * len(prefix) + encoded.offsets[:available] + [(0, 0)]
    attention = [1] * len(tokens)
    padding = max_length - len(tokens)
    tokens += [1] * padding
    attention += [0] * padding
    offsets += [(0, 0)] * padding
    return (torch.tensor(tokens, dtype=torch.long),
            torch.tensor(attention, dtype=torch.long),
            torch.tensor(offsets, dtype=torch.long))


class TweetDataset(Dataset):
    def __init__(self, data_path: str, subset: int, config: Config, is_training: bool = True):
        self.data = read_data(data_path, labeled=is_training)
        self.is_training = is_training
        if is_training:
            if 'subset' not in self.data:
                raise ValueError('Training/evaluation requires the processed CSV with a subset column')
            if subset >= 0:
                self.data = self.data[self.data['subset'] != subset]
            else:
                self.data = self.data[self.data['subset'] == abs(subset) - 1]
        self.data = self.data.reset_index(drop=True)
        self.config = config
        self.tokenizer = load_tokenizer(config)

    def __len__(self):
        return len(self.data)

    def _encode_text(self, idx):
        row = self.data.iloc[idx]
        return encode_text(self.tokenizer, row.text, row.sentiment, self.config.data.max_length)

    def _compute_labels(self, idx, offsets):
        row = self.data.iloc[idx]
        text = normalize_text(row.text)
        selected = normalize_text(row.selected_text).strip()
        start = text.find(selected)
        if start < 0:
            raise ValueError(f'Selected text is not a substring for textID={row.textID}')
        end = start + len(selected)
        target = [i for i, (a, b) in enumerate(offsets.tolist()) if b > start and a < end and b > a]
        if not target or int(offsets[target[-1], 1]) < end:
            raise ValueError(f'Selected span was truncated for textID={row.textID}; increase max_length')
        return target[0], target[-1]

    def __getitem__(self, index):
        tokens, masks, offsets = self._encode_text(index)
        text_mask = offsets[:, 1] > offsets[:, 0]
        if not text_mask.any():
            # Empty test rows retain their IDs; decoding returns an empty string.
            text_mask[0] = True
        item = {'tokens': tokens, 'masks': masks, 'text_mask': text_mask}
        if self.is_training:
            start, end = self._compute_labels(index, offsets)
            auxiliary = torch.zeros(self.config.data.max_length)
            auxiliary[start:end + 1] = 1
            item.update(start_idx=torch.tensor(start), end_idx=torch.tensor(end), aux_label=auxiliary)
        return item


def create_data_loaders(config: Config, fold: int):
    train = TweetDataset(config.data.train_path, fold, config)
    valid = TweetDataset(config.data.train_path, -fold - 1, config)
    if not len(train) or not len(valid):
        raise ValueError(f'Fold {fold} has an empty training or validation split')
    kwargs = dict(batch_size=config.data.batch_size, num_workers=config.data.num_workers, drop_last=False)
    return DataLoader(train, shuffle=True, **kwargs), DataLoader(valid, shuffle=False, **kwargs)


def create_submission_dataset(config: Config, test_path: str):
    dataset = TweetDataset(test_path, 0, config, is_training=False)
    return DataLoader(dataset, batch_size=config.data.batch_size,
                      num_workers=config.data.num_workers, shuffle=False, drop_last=False)
