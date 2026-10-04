from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from ..utils.config import Config


class DataPreprocessor:
    def __init__(self, config=None):
        self.config = config or Config()
        self.seed = self.config.SEED

    @staticmethod
    def normalize(data, source, lang='en', labeled=True):
        data = data.rename(columns={'content': 'comment_text'}).copy()
        required = {'id', 'comment_text'} | ({'toxic'} if labeled else set())
        if not required.issubset(data.columns):
            raise ValueError(f'{source}: missing columns {sorted(required - set(data.columns))}')
        data['comment_text'] = data['comment_text'].fillna('').astype(str)
        if 'source' not in data:
            data['source'] = source
        if 'lang' not in data:
            data['lang'] = lang
        if labeled:
            data['toxic'] = pd.to_numeric(data['toxic'], errors='raise')
            if not data['toxic'].between(0, 1).all():
                raise ValueError(f'{source}: toxic labels must be in [0, 1]')
        return data

    def prepare_english_data(self, data_dir):
        root = Path(data_dir)
        data1 = self.normalize(pd.read_csv(root / 'raw/jigsaw-toxic-comment-train.csv'), '2020-train')
        data2 = self.normalize(pd.read_csv(root / 'raw/jigsaw-unintended-bias-train.csv'), '2019-train')
        # Preserve the original positive-annotation filter and target threshold.
        severity = data2.get('severe_toxicity', 0)
        data2 = data2.loc[(data2['toxic'] + severity) > 0].copy()
        data2['toxic'] = (data2['toxic'] >= 0.5).astype(int)
        parts = [data1, data2]
        extra = root / 'raw/extra_english.csv'
        if extra.exists():
            parts.append(self.normalize(pd.read_csv(extra), 'extra-english'))
        data = pd.concat(parts, ignore_index=True)
        data = data.drop_duplicates('comment_text').reset_index(drop=True)
        train, valid = train_test_split(data, test_size=0.2, random_state=self.seed,
                                       stratify=(data['toxic'] >= 0.5).astype(int))
        for name, frame in [('train', train), ('valid', valid)]:
            path = root / f'process/english/{name}_english.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
        return train, valid

    def prepare_foreign_data(self, data_dir):
        root = Path(data_dir)
        # The competition validation file, not the English unintended-bias data.
        valid = self.normalize(pd.read_csv(root / 'raw/validation.csv'), 'validation', lang='unknown')
        valid['toxic'] = (valid['toxic'] >= 0.5).astype(int)
        valid['original'] = 1
        target = root / 'process/foreign'
        target.mkdir(parents=True, exist_ok=True)
        valid.to_csv(target / 'valid_foreign.csv', index=False)
        test = self.normalize(pd.read_csv(root / 'raw/test.csv', dtype={'id': str}),
                              'test', lang='unknown', labeled=False)
        test = test.drop(columns=['toxic'], errors='ignore')
        test.to_csv(target / 'test_foreign.csv', index=False)
        # Use supplied translations when available; otherwise score original text.
        english_test = root / 'raw/test_english.csv'
        translated = (self.normalize(pd.read_csv(english_test, dtype={'id': str}), 'test', labeled=False)
                      if english_test.exists() else test)
        translated = translated.drop(columns=['toxic'], errors='ignore')
        translated.to_csv(root / 'process/english/test_english.csv', index=False)
        train_path = root / 'raw/train_foreign.csv'
        train = None
        if train_path.exists():
            train = self.normalize(pd.read_csv(train_path), 'translation', lang='unknown')
            train = train.loc[~train['comment_text'].isin(valid['comment_text'])]
            train.to_csv(target / 'train_foreign.csv', index=False)
        return train, valid

    def create_pseudo_labels(self, data_dir):
        """Historical name: preserve toxicity labels and attach domain weights."""
        root = Path(data_dir) / 'process'
        frames = [pd.read_csv(root / 'english/adverse.csv')]
        for name in ['foreign', 'subtitle']:
            path = root / name / 'adverse.csv'
            if path.exists() and (root / name / ('train_foreign.csv' if name == 'foreign' else 'subtitle.csv')).exists():
                frames.append(pd.read_csv(path))
        data = pd.concat(frames, ignore_index=True)
        data = data.loc[data['toxic'].notna()].copy()
        data['weight'] = data['score'].clip(0, 1)
        path = root / 'pseudo/train_combine.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        return data

    def process_all_data(self, data_dir):
        self.prepare_english_data(data_dir)
        self.prepare_foreign_data(data_dir)
        root = Path(data_dir)
        subtitle = root / 'raw/subtitle.csv'
        if subtitle.exists():
            data = self.normalize(pd.read_csv(subtitle), 'subtitle', lang='unknown')
            target = root / 'process/subtitle/subtitle.csv'
            target.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(target, index=False)
