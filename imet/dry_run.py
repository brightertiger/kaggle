#!/usr/bin/env python3
"""Exercise the complete iMet pipeline on generated images, offline and on CPU."""
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.config import Config
from src.models import load_model_checkpoint, ModelFactory
from src.pipeline import IMetPipeline
from src.scorer import ModelScorer, SubmissionGenerator

ROOT = Path(__file__).resolve().parent


def create_sample(data_path):
    rng = np.random.default_rng(2017)
    for split in ('train', 'test'):
        (data_path / split).mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(12):
        image_id = f'train_{i:03d}'
        # RGB PNG files of varying size, with space-separated attribute ids.
        pixels = rng.integers(0, 256, (40 + i % 3, 44, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(data_path / 'train' / f'{image_id}.png')
        rows.append({'id': image_id, 'attribute_ids': f'{i % 2} 2'})
    pd.DataFrame(rows).to_csv(data_path / 'train.csv', index=False)
    # Preserve the full production output vocabulary, even for a tiny dataset.
    pd.DataFrame({'attribute_id': range(1103),
                  'attribute_name': [f'tag::synthetic_{i}' for i in range(1103)]}).to_csv(
                      data_path / 'labels.csv', index=False)
    ids = ['test_002', 'test_000', 'test_001']
    for image_id in ids:
        pixels = rng.integers(0, 256, (42, 46, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(data_path / 'test' / f'{image_id}.png')
    pd.DataFrame({'id': ids, 'attribute_ids': [''] * len(ids)}).to_csv(
        data_path / 'sample_submission.csv', index=False)
    return ids


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    data_path = ROOT / 'sample_data'
    ids = create_sample(data_path)
    config = Config(data_path=str(data_path), output_path=str(ROOT / 'dry_run_output'),
                    model_name='resnext_tiny', pretrained=False, device='cpu',
                    batch_size=5, epochs=1, image_size=32, num_folds=2,
                    num_workers=0)
    # Fail immediately if a model implementation unexpectedly tries to download.
    with patch('torch.hub.download_url_to_file', side_effect=AssertionError('Downloads forbidden')),\
         patch('torch.utils.model_zoo.load_url', side_effect=AssertionError('Downloads forbidden')):
        pipeline = IMetPipeline(config)
        results = pipeline.run_complete_pipeline()
        assert set(results) == {1, 2}
        for fold in results:
            assert np.isfinite(list(results[fold].values())).all()
            model = ModelFactory.create_model(config)
            for stage in ('stage_1', 'stage_2'):
                info = load_model_checkpoint(model, config.get_model_path(fold, stage))
                assert np.isfinite(info['loss'])
            scores = pd.read_csv(config.get_score_path(fold))
            assert scores['id'].tolist() == ids
            assert scores.shape == (len(ids), config.num_classes + 1)
            assert np.isfinite(scores.iloc[:, 1:].to_numpy()).all()
        # Validation scoring is separate from the test-set submission path.
        scorer = ModelScorer(config)
        validation = scorer.score_fold(1, config.get_model_path(1),
                                      str(ROOT / 'dry_run_output' / 'validation.csv.gz'))
        assert len(validation) == 6
        score_files = [config.get_score_path(fold) for fold in results]
        generator = SubmissionGenerator(config)
        generator.create_weighted_submission(score_files, [3, 1],
                                             config.get_submission_path('weighted_submission'))
    for name in ('submission', 'weighted_submission'):
        submission = pd.read_csv(config.get_submission_path(name), keep_default_na=False)
        assert submission.columns.tolist() == ['id', 'attribute_ids']
        assert submission['id'].tolist() == ids
        for labels in submission['attribute_ids']:
            values = [int(label) for label in labels.split()]
            assert 1 <= len(values) <= config.top_k
            assert all(0 <= label < config.num_classes for label in values)
    print('DRY RUN PASSED: 12 training images, 2 folds, frozen-head + fine-tuning, '
          'validation, test inference, mean/weighted ensembles, 3 submission rows. '
          'CPU only; random SE-ResNeXt; no downloads; no pipeline stages skipped.')


if __name__ == '__main__':
    main()
