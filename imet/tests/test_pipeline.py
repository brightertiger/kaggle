"""Regression coverage for migration failures with small local fixtures."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.config import Config
from src.data_utils import DataPreprocessor, create_data_loaders
from src.models import ModelFactory, load_model_checkpoint
from src.pipeline import IMetPipeline
from src.scorer import SubmissionGenerator
from src.trainer import MetricsCalculator, ModelTrainer

ROOT = Path(__file__).resolve().parents[1]


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = Config(data_path=str(self.root), output_path=str(self.root / 'output'),
                             model_name='resnext_tiny', pretrained=False, device='cpu',
                             num_classes=3, num_folds=2, num_workers=0, image_size=32)
        torch.set_num_threads(1)

    def test_cli_aliases_refresh_paths(self):
        self.config.update_from_args(argparse.Namespace(
            model='resnext50', lr=0.003, folds=3, output_path=str(self.root / 'new')))
        self.assertEqual(self.config.model_name, 'resnext50')
        self.assertEqual(self.config.learning_rate, 0.003)
        self.assertEqual(self.config.num_folds, 3)
        self.assertTrue(self.config.get_model_path(1).startswith(str(self.root / 'new')))

    def test_preprocess_does_not_construct_backbone(self):
        with patch('src.scorer.ModelFactory.create_model', side_effect=AssertionError('Model created')):
            pipeline = IMetPipeline(self.config)
            self.assertIsNone(pipeline.scorer._scorer)

    def test_zip_long_tail_filter_and_single_label_strings(self):
        data = pd.DataFrame({'id': [f'{i:03d}' for i in range(6)],
                             'attribute_ids': ['0 2', '0', '0', '1 2', '1', '1']})
        data.to_csv(self.config.train_csv_path, index=False,
                    compression={'method': 'zip', 'archive_name': 'train.csv'})
        pd.DataFrame({'intent': [2], 'percent': [0.1], 'total': [2]}).to_csv(
            self.config.subset_csv_path, index=False)
        folds = DataPreprocessor(self.config).create_folds()
        self.assertEqual(folds['id'].iloc[0], '000')
        self.assertEqual(set(folds['attribute_ids']), {'0', '1'})
        self.assertEqual(folds['fold'].value_counts().tolist(), [3, 3])

    def test_loaders_keep_partial_batches_and_fail_on_missing_images(self):
        (self.root / 'train').mkdir()
        data = pd.DataFrame({'id': [f'{i:03d}' for i in range(6)],
                             'attribute_ids': ['0'] * 6, 'fold': [1] * 3 + [2] * 3})
        data.to_csv(self.config.folds_csv_path, index=False)
        for image_id in data['id']:
            Image.new('RGB', (32, 32)).save(self.root / 'train' / f'{image_id}.png')
        self.config.batch_size = 4
        train, valid = create_data_loaders(self.config, 1)
        batches = list(valid)
        self.assertEqual([len(b['idx']) for b in batches], [2, 1])
        self.assertEqual(batches[-1]['label'].shape, (1, 3))
        self.assertEqual(sum(len(b['idx']) for b in train), 3)
        (self.root / 'train' / '000.png').unlink()
        with self.assertRaises(RuntimeError):
            valid.dataset[0]

    def test_f2_uses_binary_truth_and_clamps_top_k(self):
        metric = MetricsCalculator(self.config)
        truth = np.array([[0.9, 0.0001, 0.0001], [0.0001, 0.9, 0.9]])
        probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.8]])
        self.assertEqual(metric.calculate_fbeta_score(truth, probabilities), 1.0)

    def test_freezing_and_zero_score_checkpoint(self):
        model = ModelFactory.create_model(self.config)
        model.freeze()
        model.train()
        self.assertTrue(model.backbone.last_linear.weight.requires_grad)
        self.assertFalse(model.backbone.layer0.conv1.weight.requires_grad)
        self.assertFalse(model.backbone.layer0.bn1.training)
        trainer = ModelTrainer(self.config)
        metric = {'loss': 1.0, 'best_fbeta': 0.0}
        # Real optimizer step preserves scheduler ordering in this focused test.
        def train_epoch(model, loader, optimizer, loss_fn):
            optimizer.zero_grad()
            model.backbone.last_linear.weight.sum().backward()
            optimizer.step()
            return metric
        path = self.config.get_model_path(1, 'stage_1')
        with patch.object(trainer, 'train_epoch', side_effect=train_epoch), \
             patch.object(trainer, 'validate_epoch', return_value=metric):
            result = trainer.train_stage(model, None, None, stage=1, epochs=1, checkpoint_path=path)
        self.assertEqual(result['best_fbeta'], 0.0)
        self.assertEqual(load_model_checkpoint(model, path)['metric'], 0.0)
        model.unfreeze_backbone()
        model.train()
        self.assertTrue(model.backbone.layer0.conv1.weight.requires_grad)
        self.assertTrue(model.backbone.layer0.bn1.training)

    def test_ensemble_aligns_ids_and_columns_and_rejects_validation_ids(self):
        pd.DataFrame({'id': ['002', '001'], 'attribute_ids': ['', '']}).to_csv(
            self.config.sample_submission_path, index=False)
        first = pd.DataFrame({'id': ['001', '002'], 'scr_0': [0.9, 0.01],
                              'scr_1': [0.01, 0.9], 'scr_2': [0.01, 0.01]})
        second = first.iloc[::-1][['id', 'scr_2', 'scr_1', 'scr_0']]
        files = [str(self.root / f'scores_{i}.csv.gz') for i in range(2)]
        first.to_csv(files[0], index=False)
        second.to_csv(files[1], index=False)
        generator = SubmissionGenerator(self.config)
        result = generator.create_weighted_submission(files, [3, 1], str(self.root / 'submission.csv'))
        self.assertEqual(result['id'].tolist(), ['002', '001'])
        self.assertEqual(result['attribute_ids'].tolist(), ['1', '0'])
        for weights in ([0, 0], [-1, 2], [float('nan'), 1]):
            with self.assertRaises(ValueError):
                generator.create_weighted_submission(files, weights, str(self.root / 'bad.csv'))
        first.loc[0, 'id'] = 'train_image'
        first.to_csv(files[0], index=False)
        with self.assertRaises(ValueError):
            generator.create_submission(files, str(self.root / 'bad.csv'))


if __name__ == '__main__':
    unittest.main()
