import os
import torch
import pandas as pd
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from typing import Tuple, List
from .config import Config
from .models import TweetSentimentModel

class TweetEvaluator:
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = self._load_tokenizer()
    
    def _load_tokenizer(self) -> ByteLevelBPETokenizer:
        params = {
            'vocab_file': self.config.data.vocab_file,
            'merges_file': self.config.data.merges_file,
            'lowercase': True,
            'add_prefix_space': True
        }
        return ByteLevelBPETokenizer(**params)
    
    def jaccard_score(self, str1: str, str2: str) -> float:
        try:
            a = set(str1.lower().split())
            b = set(str2.lower().split())
            c = a.intersection(b)
            return float(len(c)) / (len(a) + len(b) - len(c))
        except:
            return -1.0
    
    def get_offsets(self, text: str, sentiment: str) -> List[Tuple[int, int]]:
        text = text.lower()
        sentiment = sentiment.lower().strip()
        text = " " + " ".join(text.split())
        
        encodes = self.tokenizer.encode(text)
        sentiment_ids = self.tokenizer.encode(sentiment).ids
        
        tokens = [0] + sentiment_ids + [2, 2] + encodes.ids + [2]
        offsets = [(0, 0)] * 4 + encodes.offsets + [(0, 0)]
        
        padding = self.config.data.max_length - len(tokens)
        if padding > 0:
            offsets += [(0, 0)] * padding
            
        return offsets
    
    def extract_selected_text(self, text: str, start_idx: int, end_idx: int, offsets: List[Tuple[int, int]]) -> str:
        selected_text = ""
        for ix in range(start_idx, end_idx + 1):
            if ix < len(offsets):
                selected_text += text[offsets[ix][0]: offsets[ix][1]]
                if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
                    selected_text += " "
        return selected_text
    
    def compute_score(self, text: str, sentiment: str, start_idx: int, end_idx: int, 
                     start_pred: int, end_pred: int) -> float:
        offsets = self.get_offsets(text, sentiment)
        
        if start_pred > end_pred:
            pred_text = text
        else:
            pred_text = self.extract_selected_text(text, start_pred, end_pred, offsets)
        
        true_text = self.extract_selected_text(text, start_idx, end_idx, offsets)
        
        return self.jaccard_score(true_text, pred_text)
    
    def evaluate_fold(self, fold: int, valid_data: pd.DataFrame, predictions: pd.DataFrame) -> float:
        evaluator = TweetEvaluator(self.config)
        
        data = valid_data.join(predictions, how='inner')
        data['score'] = data.apply(
            lambda x: evaluator.compute_score(
                x['text'], x['sentiment'], 
                x['start_idx'], x['end_idx'],
                x['start_pred'], x['end_pred']
            ), axis=1
        )
        
        return data['score'].mean()

class TweetScorer:
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    
    def predict_fold(self, fold: int, data_loader: torch.utils.data.DataLoader) -> pd.DataFrame:
        model = TweetSentimentModel(self.config).to(self.device)
        
        checkpoint_path = os.path.join(self.config.data.model_path, f'model_fold_{fold}.pt')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        
        start_preds = []
        end_preds = []
        start_idxs = []
        end_idxs = []
        
        with torch.no_grad():
            for batch in data_loader:
                for key, value in batch.items():
                    batch[key] = value.to(self.device)
                
                start_logits, end_logits, _ = model(batch['tokens'], batch['masks'])
                
                start_pred = start_logits.argmax(dim=1)
                end_pred = end_logits.argmax(dim=1)
                
                start_preds.extend(start_pred.cpu().numpy())
                end_preds.extend(end_pred.cpu().numpy())
                start_idxs.extend(batch['start_idx'].cpu().numpy())
                end_idxs.extend(batch['end_idx'].cpu().numpy())
                
                for key, value in batch.items():
                    batch[key] = value.to('cpu')
        
        results = pd.DataFrame({
            'start_idx': start_idxs,
            'end_idx': end_idxs,
            'start_pred': start_preds,
            'end_pred': end_preds
        })
        
        model = model.cpu()
        del model
        torch.cuda.empty_cache()
        
        return results
