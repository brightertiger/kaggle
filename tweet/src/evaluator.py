"""Fold inference and word-set Jaccard evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import Config
from .data_utils import encode_text, load_tokenizer, normalize_text
from .models import TweetSentimentModel
from .trainer import resolve_device


class TweetEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.config)
        return self._tokenizer

    @staticmethod
    def jaccard_score(str1, str2):
        a, b = set(str1.lower().split()), set(str2.lower().split())
        union = a | b
        return len(a & b) / len(union) if union else 1.0

    def get_offsets(self, text, sentiment):
        return encode_text(self.tokenizer, text, sentiment, self.config.data.max_length)[2].tolist()

    def extract_selected_text(self, text, start_idx, end_idx, offsets):
        text = normalize_text(text)
        start, end = int(start_idx), int(end_idx)
        if not 0 <= start <= end < len(offsets):
            return text.strip()
        a, b = offsets[start][0], offsets[end][1]
        if offsets[start][1] <= a or b <= offsets[end][0] or b <= a:
            return text.strip()
        return text[a:b].strip() or text.strip()

    def compute_score(self, text, sentiment, start_idx, end_idx, start_pred, end_pred):
        offsets = self.get_offsets(text, sentiment)
        truth = self.extract_selected_text(text, start_idx, end_idx, offsets)
        pred = self.extract_selected_text(text, start_pred, end_pred, offsets)
        return self.jaccard_score(truth, pred)

    def evaluate_fold(self, fold, valid_data, predictions):
        if len(valid_data) != len(predictions) or not len(valid_data):
            raise ValueError(f'Fold {fold}: validation rows and predictions must align and be nonempty')
        data = valid_data.reset_index(drop=True).join(predictions.reset_index(drop=True))
        scores = []
        for row in data.itertuples():
            offsets = self.get_offsets(row.text, row.sentiment)
            pred = self.extract_selected_text(row.text, row.start_pred, row.end_pred, offsets)
            # Compare with the actual annotated string, not its BPE reconstruction.
            scores.append(self.jaccard_score(row.selected_text, pred))
        return float(np.mean(scores))


class TweetScorer:
    def __init__(self, config: Config):
        self.config = config
        self.device = resolve_device(config)

    def predict_fold(self, fold, data_loader):
        path = Path(self.config.data.model_path) / f'model_fold_{fold}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = TweetSentimentModel(self.config, encoder_config=checkpoint['encoder_config']).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        results = {'start_pred': [], 'end_pred': []}
        if data_loader.dataset.is_training:
            results.update(start_idx=[], end_idx=[])
        with torch.no_grad():
            for batch in data_loader:
                start, end, _ = model(batch['tokens'].to(self.device), batch['masks'].to(self.device),
                                      batch['text_mask'].to(self.device))
                results['start_pred'].extend(start.argmax(dim=1).cpu().tolist())
                results['end_pred'].extend(end.argmax(dim=1).cpu().tolist())
                if 'start_idx' in results:
                    results['start_idx'].extend(batch['start_idx'].tolist())
                    results['end_idx'].extend(batch['end_idx'].tolist())
        return pd.DataFrame(results)
