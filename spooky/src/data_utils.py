"""CSV loading and ID-checked assembly of the stacking features."""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .config import Config


class DataLoader:
    def __init__(self, data_dir=None, config=None):
        self.config = config or Config()
        self.data_dir = Path(data_dir or self.config.DATA_DIR)

    def _load(self, filename, training=False):
        path = self.data_dir / filename
        df = pd.read_csv(path, dtype={'id': str}, keep_default_na=False)
        required = {'id', 'text', 'author'} if training else {'id', 'text'}
        if not required.issubset(df.columns):
            raise ValueError(f'{path} requires columns {sorted(required)}')
        if df.empty or df['id'].eq('').any() or not df['id'].is_unique:
            raise ValueError(f'{path} must contain rows with unique, nonempty IDs')
        df['text'] = df['text'].fillna('').astype(str)
        return df

    def load_train_data(self):
        return self._load(self.config.TRAIN_FILE.name, training=True)

    def load_test_data(self):
        return self._load(self.config.TEST_FILE.name)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.load_train_data(), self.load_test_data()

    def encode_authors(self, df, author_column='author'):
        df = df.copy()
        unknown = set(df[author_column]) - set(self.config.AUTHOR_MAP)
        if unknown:
            raise ValueError(f'Unknown author labels: {unknown}')
        df[author_column] = df[author_column].map(self.config.AUTHOR_MAP)
        return df


class FeatureMerger:
    def __init__(self, score_dir=None, config=None):
        self.config = config or Config()
        self.score_dir = Path(score_dir or self.config.SCORE_DIR)

    def _join(self, base, filename):
        path = self.score_dir / filename
        features = pd.read_csv(path, dtype={'id': str}, keep_default_na=False)
        if 'id' not in features or not features['id'].is_unique:
            raise ValueError(f'{path} must contain unique IDs; rerun its pipeline step')
        if set(features['id']) != set(base['id']):
            raise ValueError(f'{path} IDs do not match the current dataset')
        features = features.drop(columns=['text', 'author'], errors='ignore')
        overlap = (set(features) & set(base)) - {'id'}
        if overlap:
            raise ValueError(f'Duplicate feature columns in {path}: {overlap}')
        return base.merge(features, on='id', how='left', sort=False, validate='one_to_one')

    def merge_all_features(self, train_data, test_data):
        train, test = train_data.copy(), test_data.copy()
        # Fail clearly on missing stages instead of silently training a different stack.
        for train_file, test_file in [
            (self.config.TRAIN_TEXT_FEATS, self.config.TEST_TEXT_FEATS),
            (self.config.TRAIN_NB_SCORE, self.config.TEST_NB_SCORE),
            (self.config.TRAIN_NB_FEATS, self.config.TEST_NB_FEATS),
            (self.config.TRAIN_NN_SCORE, self.config.TEST_NN_SCORE),
            (self.config.TRAIN_LSTM_SCORE, self.config.TEST_LSTM_SCORE),
        ]:
            train = self._join(train, train_file)
            test = self._join(test, test_file)
        for frame in (train, test):
            frame['nn_prob'] = frame[[f'simple_{i}' for i in range(self.config.NUM_CLASSES)]].max(axis=1)
            frame['lstm_prob'] = frame[[f'lstm_{i}' for i in range(self.config.NUM_CLASSES)]].max(axis=1)
            frame['agree'] = (frame['keras'] == frame['lstm']).astype(float)
        return train, test

    def prepare_final_data(self, train_data, test_data):
        train = train_data.drop(columns=['text'])
        test = test_data.drop(columns=['text'])
        features = train.columns.drop(['id', 'author'])
        if set(features) != set(test.columns.drop('id')):
            raise ValueError('Training and test feature schemas differ')
        test = test[['id', *features]]
        for frame in (train[features], test[features]):
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError('Stacking features must be finite numeric values')
        return train, test


class DataProcessor:
    def __init__(self, data_dir=None, score_dir=None, config=None):
        self.data_loader = DataLoader(data_dir, config)
        self.feature_merger = FeatureMerger(score_dir, config)

    def process_data(self):
        train, test = self.data_loader.load_data()
        train = self.data_loader.encode_authors(train)
        train, test = self.feature_merger.merge_all_features(train, test)
        return self.feature_merger.prepare_final_data(train, test)
