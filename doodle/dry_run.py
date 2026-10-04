#!/usr/bin/env python3
"""Exercise the real CLI on synthetic Quick, Draw! CSVs, entirely on CPU."""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from main import main
from src import DoodleDataset

ROOT = Path(__file__).resolve().parent
CATEGORIES = ['airplane', 'apple', 'ice cream', 'zigzag']


def drawing_for(category, sample):
    """Simplified-format strokes: a list of [x coordinates, y coordinates]."""
    offset = sample % 5
    shapes = [
        [[[30, 128, 225, 128, 30], [160, 35, 160, 120, 160]]],
        [[[60, 40, 60, 128, 195, 215, 195, 128, 60],
          [70, 140, 210, 225, 210, 140, 70, 50, 70]]],
        [[[55, 128, 200, 55], [110, 225, 110, 110]],
         [[55, 70, 128, 185, 200], [110, 60, 40, 60, 110]]],
        [[[30, 225, 30, 225], [40, 100, 160, 220]]],
    ]
    return json.dumps([[[x + offset for x in xs], ys] for xs, ys in shapes[category]])


def generate_sample():
    sample_path = ROOT / 'sample_data'
    source_path = sample_path / 'train_simplified'
    source_path.mkdir(parents=True, exist_ok=True)
    for category, word in enumerate(CATEGORIES):
        rows = [dict(
            countrycode='US', drawing=drawing_for(category, sample),
            key_id=str(900000000000000001 + category * 100 + sample),
            recognized=sample % 2 == 0, timestamp='2017-03-01 12:00:00 UTC', word=word,
        ) for sample in range(12)]
        pd.DataFrame(rows).to_csv(source_path / f'{word}.csv', index=False)
    test = pd.DataFrame([
        dict(countrycode='US', drawing=drawing_for(i % len(CATEGORIES), i),
             key_id=str(900000000000010001 + i)) for i in range(5)
    ])
    test_path = sample_path / 'test_simplified.csv'
    test.to_csv(test_path, index=False)
    pd.DataFrame({'key_id': test.key_id, 'word': 'airplane apple ice_cream'}).to_csv(
        sample_path / 'sample_submission.csv', index=False,
    )
    return source_path, test_path, test


def run():
    torch.set_num_threads(1)
    source_path, test_path, test = generate_sample()
    output_path = ROOT / 'dry_run_output'
    args = [
        'main.py', '--step', 'all', '--model', 'resnet18',
        '--source-data', str(source_path), '--test-data', str(test_path),
        '--data-dir', str(output_path), '--device', 'cpu', '--no-pretrained',
        '--epochs', '1', '--batch-size', '6', '--image-size', '32',
        '--num-workers', '0', '--train-ratio', '0.75',
    ]
    # Fail explicitly if a future change accidentally requests pretrained weights.
    with patch('sys.argv', args), patch('torch.hub.download_url_to_file',
                                      side_effect=AssertionError('Dry run must not download weights')):
        main()
    train = pd.read_csv(output_path / 'train/train.csv', dtype={'key_id': str})
    valid = pd.read_csv(output_path / 'valid/valid.csv', dtype={'key_id': str})
    assert set(train.key_id).isdisjoint(valid.key_id)
    assert set(train.word) == set(valid.word) == set(CATEGORIES)
    raster = DoodleDataset(train, CATEGORIES, image_size=32, is_training=False)[0]['image']
    assert raster.shape == (1, 32, 32) and torch.isfinite(raster).all()
    assert raster.max() > 0 and raster.min() >= 0 and raster.max() <= 1
    submission_path = output_path / 'submit/resnet18_submission.csv'
    submission = pd.read_csv(submission_path, dtype={'key_id': str})
    assert submission.columns.tolist() == ['key_id', 'word']
    assert submission.key_id.tolist() == test.key_id.tolist()
    allowed = {word.replace(' ', '_') for word in CATEGORIES}
    assert all(len(words.split()) == len(set(words.split())) == 3
               and set(words.split()) <= allowed for words in submission.word)
    assert (output_path / 'model/resnet18/resnet18_best.pth').exists()
    print(f'DRY RUN PASS: {len(train)} train / {len(valid)} validation / {len(test)} test rows; '
          'stroke rasterization, ResNet18 training, checkpoint reload, prediction and submission.')
    print(f'CPU only; random initialization; no downloads; no stages skipped. Output: {submission_path}')


if __name__ == '__main__':
    run()
