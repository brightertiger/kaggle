"""Cross-validation training and fold-averaged GAP predictions."""
import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
from transformers import BertConfig, BertTokenizer

from .config import Config
from .models import PronounResolutionModel
from .data_utils import PronounDataset, create_data_loaders, collate_fn, load_and_process_data, read_gap
from .trainer import PronounTrainer
from .optimizer import AdaBound, CosineLR
from .feature_engineering import FeatureExtractor


class PronounResolutionPipeline:
    def __init__(self, config: Config, device: str = None):
        self.config = config
        requested = device or config.device
        if requested.startswith('cuda') and not torch.cuda.is_available():
            warnings.warn('CUDA unavailable; using CPU', stacklevel=2)
            requested = 'cpu'
        self.device = torch.device(requested)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.feature_extractor = FeatureExtractor(config.data.spacy_model)
        self.tokenizer = BertTokenizer.from_pretrained(
            config.model.pretrained_model, do_lower_case=True,
            local_files_only=config.model.random_init)
        if self.tokenizer.pad_token_id != 0:
            raise ValueError('This BERT pipeline expects padding token ID 0')

    def _new_model(self):
        cfg = self.config.model
        bert_config = None
        if cfg.random_init:
            bert_config = BertConfig(
                vocab_size=len(self.tokenizer), hidden_size=cfg.hidden_size,
                num_hidden_layers=cfg.num_hidden_layers,
                num_attention_heads=cfg.num_attention_heads,
                intermediate_size=cfg.intermediate_size,
                max_position_embeddings=cfg.max_length,
                hidden_dropout_prob=cfg.dropout, attention_probs_dropout_prob=cfg.dropout)
        model = PronounResolutionModel(cfg.pretrained_model, cfg.hidden_size, cfg.dropout,
                                       bert_config=bert_config, freeze_layers=cfg.freeze_layers)
        if cfg.max_length > model.bert_encoder.bert.config.max_position_embeddings:
            raise ValueError('max_length exceeds the BERT positional embedding limit')
        return model.to(self.device)

    def _features(self, data):
        data = self.feature_extractor.extract_basic_features(data)
        return self.feature_extractor.extract_linguistic_features(data)

    def prepare_data(self, train_path, val_path):
        train_data, val_data = load_and_process_data(train_path, val_path)
        combined = pd.concat([train_data, val_data], ignore_index=True)
        if combined['ID'].duplicated().any():
            raise ValueError('Training source files contain overlapping IDs')
        if len(combined) < self.config.data.n_folds:
            raise ValueError('There must be at least one example per fold')
        # Preserve the row-modulo split used by the migrated solution.
        combined['fold'] = np.arange(len(combined)) % self.config.data.n_folds + 1
        return self._features(combined)

    def _add_tags_to_text(self, data):
        def tag(row):
            text = row['Text']
            markers = [(int(row[f'{name}-offset']), marker)
                       for name, marker in [('A', '[A]'), ('B', '[B]'), ('Pronoun', '[P]')]]
            # Insert from right to left: offsets stay tied to the untouched text.
            for offset, marker in sorted(markers, reverse=True):
                text = text[:offset] + f' {marker} ' + text[offset:]
            return text
        result = data.copy()
        result['Text'] = result.apply(tag, axis=1)
        return result

    def _create_labels(self, data):
        return pd.DataFrame({'Label': np.where(data['A-coref'], 0, np.where(data['B-coref'], 1, 2))},
                            index=data.index)

    def _dataset(self, data, labeled=False):
        columns = lambda candidate: [f'dist_{candidate}'] + [f'{candidate}_{suffix}' for suffix in
                                                           ('url', 'cc', 'par', 'th', 'loc', 'cloc')]
        return PronounDataset(
            self._add_tags_to_text(data)[['Text']],
            (data[columns('a')], data[columns('b')]),
            self._create_labels(data) if labeled else None,
            self.tokenizer, max_length=self.config.model.max_length)

    def train(self):
        combined = self.prepare_data(self.config.data.train_path, self.config.data.val_path)
        histories = {}
        for fold in range(1, self.config.data.n_folds + 1):
            print(f'Training fold {fold}')
            histories[fold] = self._train_fold(combined[combined['fold'] != fold],
                                               combined[combined['fold'] == fold], fold)
        return histories

    def _train_fold(self, train_data, val_data, fold):
        loaders = create_data_loaders(self._dataset(train_data, True), self._dataset(val_data, True),
                                      self.config.model.batch_size, self.config.data.num_workers)
        model = self._new_model()
        optimizer = AdaBound(model.parameters(), lr=self.config.model.learning_rate,
                             weight_decay=self.config.model.weight_decay)
        scheduler = CosineLR(optimizer, T_max=100, T_mult=0.9, eta_min=1e-4)
        trainer = PronounTrainer(model, self.device, optimizer, scheduler)
        return trainer.train(*loaders, epochs=self.config.model.epochs,
                             save_dir=os.path.join(self.config.data.output_dir, f'fold_{fold}'))

    def predict(self, test_path=None):
        data = self._features(read_gap(test_path or self.config.data.test_path))
        loader = torch.utils.data.DataLoader(self._dataset(data),
                    batch_size=self.config.model.batch_size, shuffle=False,
                    collate_fn=collate_fn, num_workers=self.config.data.num_workers)
        predictions = []
        for fold in range(1, self.config.data.n_folds + 1):
            print(f'Predicting with fold {fold}')
            model = self._new_model()
            path = os.path.join(self.config.data.output_dir, f'fold_{fold}', 'best_model.pth')
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            fold_predictions = []
            with torch.no_grad():
                for batch in loader:
                    tokens, offsets, feature_a, feature_b, _ = [x.to(self.device) for x in batch]
                    fold_predictions.append(model(tokens, offsets, feature_a, feature_b).softmax(1).cpu().numpy())
            predictions.append(np.vstack(fold_predictions))
        submission = pd.DataFrame(np.mean(predictions, axis=0), columns=['A', 'B', 'NEITHER'])
        submission.insert(0, 'ID', data['ID'].to_numpy())
        path = os.path.join(self.config.data.output_dir, 'submission.csv')
        submission.to_csv(path, index=False)
        print(f'Predictions saved to {path}')
        return submission
