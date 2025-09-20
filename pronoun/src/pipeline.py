import os
import pandas as pd
import torch
import numpy as np
from typing import Tuple, Dict, Any
from pytorch_pretrained_bert import BertTokenizer
import torch.nn.functional as F
from tqdm import tqdm

from .config import Config
from .models import PronounResolutionModel
from .data_utils import PronounDataset, create_data_loaders
from .trainer import PronounTrainer
from .optimizer import AdaBound, CosineLR
from .feature_engineering import FeatureExtractor

class PronounResolutionPipeline:
    def __init__(self, config: Config, device: str):
        self.config = config
        self.device = torch.device(device)
        self.feature_extractor = FeatureExtractor()
        self.tokenizer = self._setup_tokenizer()
        
    def _setup_tokenizer(self) -> BertTokenizer:
        params = {
            'pretrained_model_name_or_path': self.config.model.pretrained_model,
            'do_lower_case': True,
            'never_split': ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[A]", "[B]", "[P]")
        }
        tokenizer = BertTokenizer.from_pretrained(**params)
        tokenizer.vocab["[A]"] = -1
        tokenizer.vocab["[B]"] = -1
        tokenizer.vocab["[P]"] = -1
        return tokenizer
    
    def prepare_data(self, train_path: str, val_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_data = pd.read_csv(train_path, sep='\t')
        val_data = pd.read_csv(val_path, sep='\t')
        
        combined_data = pd.concat([train_data, val_data], ignore_index=True)
        combined_data['fold'] = combined_data.index.map(lambda x: (x % 5) + 1)
        
        combined_data = self.feature_extractor.extract_basic_features(combined_data)
        combined_data = self.feature_extractor.extract_linguistic_features(combined_data)
        
        return combined_data[combined_data['fold'] != 1], combined_data[combined_data['fold'] == 1]
    
    def _add_tags_to_text(self, data: pd.DataFrame) -> pd.DataFrame:
        def replace_name_with_tag(text, name, offset, tag):
            before_text = text[:offset - 20]
            after_text = text[offset + 20:]
            replace_text = text[offset - 20:offset + 20]
            replace_text = replace_text.replace(name, tag)
            return before_text + replace_text + after_text
        
        data = data.copy()
        data['Text'] = data[['Text', 'A', 'A-offset']].apply(
            lambda x: replace_name_with_tag(*x, ' [A] '), axis=1)
        data['Text'] = data[['Text', 'B', 'B-offset']].apply(
            lambda x: replace_name_with_tag(*x, ' [B] '), axis=1)
        data['Text'] = data[['Text', 'Pronoun', 'Pronoun-offset']].apply(
            lambda x: replace_name_with_tag(*x, ' [P] '), axis=1)
        
        return data
    
    def _create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        def create_label(row):
            if row['A-coref'] == 1:
                return 0
            elif row['B-coref'] == 1:
                return 1
            else:
                return 2
        
        labels = data.apply(create_label, axis=1)
        return pd.DataFrame({'Label': labels})
    
    def train(self):
        train_data, val_data = self.prepare_data(
            self.config.data.train_path, 
            self.config.data.val_path
        )
        
        for fold in range(1, self.config.data.n_folds + 1):
            print(f"Training fold {fold}")
            
            fold_train = train_data[train_data['fold'] != fold]
            fold_val = train_data[train_data['fold'] == fold]
            
            self._train_fold(fold_train, fold_val, fold)
    
    def _train_fold(self, train_data: pd.DataFrame, val_data: pd.DataFrame, fold: int):
        train_data_tagged = self._add_tags_to_text(train_data)
        val_data_tagged = self._add_tags_to_text(val_data)
        
        train_labels = self._create_labels(train_data)
        val_labels = self._create_labels(val_data)
        
        feature_columns_a = ['dist_a', 'a_url', 'a_cc', 'a_par', 'a_th', 'a_loc', 'a_cloc']
        feature_columns_b = ['dist_b', 'b_url', 'b_cc', 'b_par', 'b_th', 'b_loc', 'b_cloc']
        
        train_features = (train_data[feature_columns_a], train_data[feature_columns_b])
        val_features = (val_data[feature_columns_a], val_data[feature_columns_b])
        
        train_dataset = PronounDataset(
            train_data_tagged[['Text']], 
            train_features, 
            train_labels, 
            self.tokenizer
        )
        
        val_dataset = PronounDataset(
            val_data_tagged[['Text']], 
            val_features, 
            val_labels, 
            self.tokenizer
        )
        
        train_loader, val_loader = create_data_loaders(
            train_dataset, 
            val_dataset, 
            self.config.model.batch_size
        )
        
        model = PronounResolutionModel(
            self.config.model.pretrained_model,
            self.config.model.hidden_size,
            self.config.model.dropout
        )
        
        optimizer = AdaBound(
            model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay
        )
        
        scheduler = CosineLR(
            optimizer,
            T_max=100,
            T_mult=0.9,
            eta_min=1e-4
        )
        
        trainer = PronounTrainer(
            model=model,
            device=self.device,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        save_dir = os.path.join(self.config.data.output_dir, f'fold_{fold}')
        os.makedirs(save_dir, exist_ok=True)
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config.model.epochs,
            save_dir=save_dir
        )
        
        print(f"Fold {fold} training completed. Best validation loss: {trainer.best_val_loss:.4f}")
    
    def predict(self, test_path: str = None):
        if test_path is None:
            test_path = self.config.data.test_path
        
        test_data = pd.read_csv(test_path, sep='\t')
        test_data = self.feature_extractor.extract_basic_features(test_data)
        test_data = self.feature_extractor.extract_linguistic_features(test_data)
        
        predictions = []
        
        for fold in range(1, self.config.data.n_folds + 1):
            print(f"Predicting with fold {fold}")
            
            test_data_tagged = self._add_tags_to_text(test_data)
            
            feature_columns_a = ['dist_a', 'a_url', 'a_cc', 'a_par', 'a_th', 'a_loc', 'a_cloc']
            feature_columns_b = ['dist_b', 'b_url', 'b_cc', 'b_par', 'b_th', 'b_loc', 'b_cloc']
            
            test_features = (test_data[feature_columns_a], test_data[feature_columns_b])
            
            test_dataset = PronounDataset(
                test_data_tagged[['Text']], 
                test_features, 
                None, 
                self.tokenizer
            )
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=50,
                shuffle=False,
                collate_fn=lambda x: self._collate_fn(x),
                num_workers=1,
                pin_memory=True,
                drop_last=False
            )
            
            model = PronounResolutionModel(
                self.config.model.pretrained_model,
                self.config.model.hidden_size,
                self.config.model.dropout
            )
            
            checkpoint_path = os.path.join(self.config.data.output_dir, f'fold_{fold}', 'best_model.pth')
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            fold_predictions = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Fold {fold}"):
                    tokens, offsets, feature_a, feature_b, _ = [x.to(self.device) for x in batch]
                    outputs = F.softmax(model(tokens, offsets, feature_a, feature_b), dim=1)
                    fold_predictions.append(outputs.cpu().numpy())
            
            predictions.append(np.vstack(fold_predictions))
        
        ensemble_predictions = np.mean(predictions, axis=0)
        
        submission = pd.DataFrame(ensemble_predictions, columns=['A', 'B', 'NEITHER'])
        submission['ID'] = [f'development-{i+1}' for i in range(len(submission))]
        
        output_path = os.path.join(self.config.data.output_dir, 'submission.csv')
        submission.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        return submission
    
    def _collate_fn(self, batch):
        from .data_utils import collate_fn
        return collate_fn(batch)
