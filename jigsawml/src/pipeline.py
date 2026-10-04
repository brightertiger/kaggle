from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils.config import Config
from .models import XLMRobertaClassifier, WeightedBCELoss
from .data import create_data_loaders, EmbeddingProcessor, DataPreprocessor
from .training import ModelTrainer, ScoringPipeline
from .utils import AdversarialGenerator, ModelEnsemble


class JigsawPipeline:
    def __init__(self, config=None, embedding_encoder=None):
        self.config = config or Config()
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        self.adversarial_generator = AdversarialGenerator(self.config)
        self.ensemble = ModelEnsemble(self.config.MODEL_DIR, self.config)
        self.embedding_processor = EmbeddingProcessor(self.config, embedding_encoder)
        self.data_preprocessor = DataPreprocessor(self.config)
        self.scoring_pipeline = ScoringPipeline(self.config.MODEL_DIR, self.config)

    def _train(self, subset, version, load_checkpoint=False):
        train_loader, valid_loader = create_data_loaders(subset, self.config)
        checkpoint = None
        if load_checkpoint:
            source_version = 1
            path = Path(self.config.MODEL_DIR) / f'version{source_version}/model_{subset}.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = XLMRobertaClassifier(self.config,
                                    backbone_config=checkpoint.get('backbone_config') if checkpoint else None)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.config.DEVICE)
        params = list(model.named_parameters())
        exc = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        groups = [
            {'params': [p for n, p in params if not any(ex in n for ex in exc)], 'weight_decay': 0.01},
            {'params': [p for n, p in params if any(ex in n for ex in exc)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(groups, lr=self.config.LR_V1 if version == 1 else self.config.LR_V2)
        scheduler = (ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6, patience=0)
                     if version == 1 else ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-7))
        trainer = ModelTrainer(
            model, train_loader, valid_loader,
            WeightedBCELoss() if version == 1 else nn.BCEWithLogitsLoss(reduction='none'),
            optimizer, scheduler, f'{self.config.MODEL_DIR}/version{version}', subset, self.config,
        )
        trainer.train(self.config.EPOCHS_V1 if version == 1 else self.config.EPOCHS_V2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_version1(self, subset, load_pretrained=False):
        """Optionally initialize from an existing version-1 checkpoint."""
        self._train(subset, 1, load_pretrained)

    def train_version2(self, subset, load_from_version1=True):
        self._train(subset, 2, load_from_version1)

    def prepare_data(self, data_dir):
        self.config.DATA_DIR = str(Path(data_dir) / 'process')
        self.data_preprocessor.process_all_data(data_dir)

    def generate_embeddings(self, data_dir):
        self.embedding_processor.process_all_datasets(Path(data_dir) / 'process')

    def generate_adversarial_data(self, data_dir):
        self.adversarial_generator.generate_all_adversarial_data(Path(data_dir) / 'process')
        self.data_preprocessor.create_pseudo_labels(data_dir)

    def create_ensemble(self):
        return self.ensemble.create_final_ensemble()

    def run_full_pipeline(self, data_dir=None, test_path=None):
        data_dir = data_dir or str(Path(self.config.DATA_DIR).parent)
        self.prepare_data(data_dir)
        self.generate_embeddings(data_dir)
        self.generate_adversarial_data(data_dir)
        for version in [1, 2]:
            for subset in range(self.config.N_FOLDS):
                print(f'Training version {version}, subset {subset}')
                self._train(subset, version, load_checkpoint=(version == 2))
        test_path = test_path or str(Path(self.config.DATA_DIR) / 'foreign/test_foreign.csv')
        self.scoring_pipeline.score_all_models(test_path)
        return self.create_ensemble()
