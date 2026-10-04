from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from .config import Config


class AdversarialGenerator:
    def __init__(self, config=None):
        self.config = config or Config()
        self.lgb_params = self.config.LGB_PARAMS
        self.use_features = self.config.USE_FEATURES

    def create_adversarial_data(self, train_path, valid_path, test_path, output_path):
        train, valid, test = [pd.read_csv(p) for p in (train_path, valid_path, test_path)]
        data = pd.concat([train, valid, test], ignore_index=True)
        # Predict domain membership, never toxicity: target-like rows get weight.
        labels = np.r_[np.zeros(len(train), dtype=int), np.ones(len(valid) + len(test), dtype=int)]
        scores = np.zeros(len(data))
        cv = StratifiedKFold(self.config.N_FOLDS, shuffle=True, random_state=self.config.SEED)
        for train_idx, valid_idx in cv.split(data, labels):
            train_matrix = lgb.Dataset(data.iloc[train_idx][self.use_features], labels[train_idx])
            valid_matrix = lgb.Dataset(data.iloc[valid_idx][self.use_features], labels[valid_idx])
            model = lgb.train(
                params=self.lgb_params, train_set=train_matrix, valid_sets=[valid_matrix],
                num_boost_round=self.config.LGB_ROUNDS,
                callbacks=[lgb.early_stopping(self.config.LGB_EARLY_STOPPING, verbose=False)],
            )
            scores[valid_idx] = model.predict(
                data.iloc[valid_idx][self.use_features],
                num_threads=self.lgb_params.get('num_threads', 1),
            )
        # Keep text and existing labels; held-out/test rows must not enter training.
        result = train.drop(columns=self.use_features).copy()
        result['score'] = scores[:len(train)]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        return result

    def generate_all_adversarial_data(self, data_dir):
        root = Path(data_dir)
        datasets = [
            ('english', 'english/train_english', 'english/valid_english', 'english/test_english'),
            ('subtitle', 'subtitle/subtitle', 'foreign/valid_foreign', 'foreign/test_foreign'),
            ('foreign', 'foreign/train_foreign', 'foreign/valid_foreign', 'foreign/test_foreign'),
        ]
        for name, train, valid, test in datasets:
            train_path = root / f'{train}_embed.csv'
            if name != 'english' and not train_path.exists():
                print(f'Skipping optional {name} domain weights: no input dataset')
                continue
            self.create_adversarial_data(train_path, root / f'{valid}_embed.csv',
                                         root / f'{test}_embed.csv', root / name / 'adverse.csv')
