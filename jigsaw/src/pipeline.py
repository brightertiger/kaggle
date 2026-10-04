"""Preprocessing, fold training, bias evaluation, and submission generation."""
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import AutoTokenizer

from .config import Config
from .data_utils import DataProcessor, DataLoader
from .models import BERTClassifier, GPTClassifier, ModelTrainer, CustomLoss
from .models import ToxicCommentDataset, prediction_scores
from .evaluation import ModelEvaluator


class JigsawPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.data_loader = DataLoader(config)
        self.evaluator = ModelEvaluator(config)
        self.models = {}
        self.predictions = {}
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        self.device = torch.device(config.device if config.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
        config.create_directories()
        print(f'Using {self.device}')

    def validate_setup(self):
        return self.config.validate_setup()

    def process_data(self):
        return self.data_processor.process_all_data()

    @staticmethod
    def _model_types(model_type):
        if model_type not in ('bert', 'gpt', 'both'):
            raise ValueError(f'Unknown model type: {model_type}')
        return ('bert', 'gpt') if model_type == 'both' else (model_type,)

    def get_tokenizer(self, model_type='bert'):
        self._model_types(model_type)
        settings = self.config.bert_config if model_type == 'bert' else self.config.gpt_config
        tokenizer = AutoTokenizer.from_pretrained(
            settings.get('tokenizer_name', settings['model_name']),
            local_files_only=self.config.random_init,
        )
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            raise ValueError('Tokenizer needs a padding or EOS token')
        return tokenizer

    def train_bert_model(self, fold):
        return self._train_model('bert', fold)

    def train_gpt_model(self, fold):
        return self._train_model('gpt', fold)

    def _train_model(self, model_type, fold):
        train_data, valid_data = self.data_loader.load_fold_data(fold)
        tokenizer = self.get_tokenizer(model_type)
        trainer = ModelTrainer(self.config, model_type)
        settings = trainer.model_config
        if settings['num_epochs'] < 1:
            raise ValueError('num_epochs must be positive')
        model_class = BERTClassifier if model_type == 'bert' else GPTClassifier
        model = model_class(self.config).to(self.device)
        train_loader, valid_loader = trainer.create_data_loaders(train_data, valid_data, tokenizer)
        steps = math.ceil(len(train_loader) / settings['gradient_accumulation_steps']) * settings['num_epochs']
        optimizer = trainer.setup_optimizer(model, steps)
        loss_fn = CustomLoss.bert_loss if model_type == 'bert' else CustomLoss.gpt_loss
        best_loss, patience = float('inf'), 0
        model_path = self.config.get_model_path(model_type, f'fold_{fold}_best_model.pt')
        predictions_path = self.config.get_output_path('predictions', f'{model_type}_fold_{fold}_predictions.csv')
        for epoch in range(settings['num_epochs']):
            train_loss = trainer.train_epoch(model, train_loader, optimizer, loss_fn, self.device)
            valid_loss, valid_predictions = trainer.validate_model(model, valid_loader, self.device)
            print(f'{model_type} fold {fold}, epoch {epoch + 1}: train={train_loss:.4f}, valid={valid_loss:.4f}')
            if not np.isfinite(valid_loss):
                raise ValueError('Non-finite validation loss')
            if valid_loss < best_loss:
                best_loss, patience = valid_loss, 0
                trainer.save_model(model, optimizer, epoch, valid_loss, model_path)
                best_predictions = valid_predictions
                best_predictions.to_csv(predictions_path, index=False)
            else:
                patience += 1
            if patience >= self.config.training_config['early_stopping_patience']:
                break
        trainer.load_model(model, model_path)
        # Retain the public single-fold return API without holding every fold on the GPU.
        return {'model': model.cpu(), 'best_loss': best_loss, 'predictions': best_predictions}

    def train_all_models(self, model_type='both'):
        selected = self._model_types(model_type)
        config_path = Path(self.config.get_model_path(filename='config.json'))
        config_path.write_text(json.dumps(self.config.to_dict(), indent=2))
        results = {}
        for fold in range(1, self.config.n_folds + 1):
            results[f'fold_{fold}'] = {}
            for name in selected:
                result = self._train_model(name, fold)
                del result['model']
                result['checkpoint'] = self.config.get_model_path(name, f'fold_{fold}_best_model.pt')
                results[f'fold_{fold}'][name] = result
        self.models = results
        return results

    def evaluate_models(self, model_type='both'):
        evaluations = {}
        for fold in range(1, self.config.n_folds + 1):
            fold_results = {}
            for name in self._model_types(model_type):
                predictions_path = Path(self.config.get_output_path('predictions', f'{name}_fold_{fold}_predictions.csv'))
                if not predictions_path.exists():
                    continue
                _, valid_data = self.data_loader.load_fold_data(fold)
                ground_truth = valid_data[['id', 'target'] + self.config.identity_columns]
                eval_path = self.config.get_output_path('evaluations', f'{name}_fold_{fold}_evaluation')
                fold_results[name] = self.evaluator.evaluate_predictions(
                    pd.read_csv(predictions_path), ground_truth, eval_path
                )
            if fold_results:
                evaluations[f'fold_{fold}'] = fold_results
        if not evaluations:
            raise FileNotFoundError('No saved validation predictions; train models first.')
        return evaluations

    def generate_test_predictions(self, model_type='bert'):
        if model_type == 'both':
            frames = [self.generate_test_predictions(name) for name in self._model_types(model_type)]
            submission = frames[0].copy()
            submission['prediction'] = np.mean([frame['prediction'].to_numpy() for frame in frames], axis=0)
        else:
            self._model_types(model_type)
            test_data = self.data_loader.load_test_data()
            tokenizer = self.get_tokenizer(model_type)
            trainer = ModelTrainer(self.config, model_type)
            loader = TorchDataLoader(
                ToxicCommentDataset(test_data, tokenizer, self.config.max_length),
                batch_size=trainer.model_config['valid_batch_size'], shuffle=False,
                num_workers=self.config.training_config['num_workers'],
            )
            fold_predictions = []
            for fold in range(1, self.config.n_folds + 1):
                model_path = self.config.get_model_path(model_type, f'fold_{fold}_best_model.pt')
                if not Path(model_path).exists():
                    raise FileNotFoundError(f'Missing fold checkpoint: {model_path}')
                model_class = BERTClassifier if model_type == 'bert' else GPTClassifier
                model = trainer.load_model(model_class(self.config), model_path).to(self.device)
                model.eval()
                scores = []
                with torch.no_grad():
                    for batch in loader:
                        outputs = model(batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device))
                        logits = outputs[0] if model_type == 'bert' else outputs
                        scores.extend(prediction_scores(logits, model_type).cpu().tolist())
                fold_predictions.append(scores)
                del model
            submission = pd.DataFrame({'id': test_data['id'], 'prediction': np.mean(fold_predictions, axis=0)})
        path = self.config.get_output_path('submissions', f'{model_type}_submission.csv')
        submission.to_csv(path, index=False)
        print(f'Submission saved to: {path}')
        return submission

    def run_full_pipeline(self, model_type='both'):
        if not self.validate_setup():
            raise RuntimeError('Setup validation failed')
        self.process_data()
        self.train_all_models(model_type)
        evaluations = self.evaluate_models(model_type)
        submission = self.generate_test_predictions(model_type)
        predictions = {model_type: submission}
        if model_type == 'both':
            predictions.update({name: pd.read_csv(self.config.get_output_path('submissions', f'{name}_submission.csv'))
                                for name in ('bert', 'gpt')})
        return {'models': self.models, 'evaluations': evaluations, 'test_predictions': predictions}
