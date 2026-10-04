#!/usr/bin/env python3
"""Generate competition-shaped data and exercise the full CPU CLI without downloads."""
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.core import Config
from src.inference.predictor import ModelPredictor

ROOT = Path(__file__).resolve().parent


def make_sample():
    raw = ROOT / 'sample_data' / 'download'
    rng = np.random.default_rng(2017)
    train_rows, test_rows, depth_rows = [], [], []
    yy, xx = np.mgrid[:101, :101]
    for split, count in [('train', 8), ('test', 3)]:
        (raw / split / 'images').mkdir(parents=True, exist_ok=True)
        if split == 'train':
            (raw / split / 'masks').mkdir(parents=True, exist_ok=True)
        for i in range(count):
            image_id = f'{split}_{i:03d}'
            mask = np.zeros((101, 101), dtype=np.uint8)
            if i % 2:
                mask[((xx - 50) ** 2 + (yy - 60) ** 2) < (25 + i) ** 2] = 255
            image = np.clip(90 + 25 * np.sin(yy / 5) + rng.normal(0, 10, (101, 101)) + mask * 0.3, 0, 255).astype(np.uint8)
            Image.fromarray(image).save(raw / split / 'images' / f'{image_id}.png')
            depth_rows.append({'id': image_id, 'z': 100 + i})
            if split == 'train':
                Image.fromarray(mask).save(raw / split / 'masks' / f'{image_id}.png')
                train_rows.append({'id': image_id, 'rle_mask': ModelPredictor._rle_encode(mask > 0)})
            else:
                test_rows.append({'id': image_id, 'rle_mask': ''})
    pd.DataFrame(train_rows).to_csv(raw / 'train.csv', index=False)
    pd.DataFrame(test_rows).to_csv(raw / 'sample_submission.csv', index=False)
    pd.DataFrame(depth_rows).to_csv(raw / 'depths.csv', index=False)
    return [row['id'] for row in test_rows]


def main():
    test_ids = make_sample()
    # Set the thread count inside the child too; avoid CPU oversubscription on tiny batches.
    command = [sys.executable, '-c',
               'import torch; torch.set_num_threads(1); from main import main; main()',
               '--mode', 'full', '--device', 'cpu', '--tiny-model', '--no-pretrained',
               '--epochs', '1', '--folds', '2', '--batch-size', '2',
               '--image-size', '25', '--padded-size', '32', '--use-tta',
               '--data-dir', str(ROOT / 'sample_data'), '--output-dir', str(ROOT / 'dry_run_output')]
    subprocess.run(command, cwd=ROOT, check=True)
    config = Config(ROOT / 'sample_data', ROOT / 'dry_run_output')
    config.TINY_MODEL = True
    submission = pd.read_csv(config.SUBMIT_DIR / 'seresnet34_tiny_submission.csv', keep_default_na=False)
    assert submission.columns.tolist() == ['id', 'rle_mask']
    assert submission.id.tolist() == test_ids
    for rle in submission.rle_mask:
        runs = np.array([int(token) for token in rle.split()], dtype=int)
        assert len(runs) % 2 == 0
        if len(runs):
            assert (runs[::2] >= 1).all() and (runs[1::2] > 0).all()
            assert (runs[::2] + runs[1::2] - 1 <= 101 * 101).all()
    for fold in (1, 2):
        checkpoint = torch.load(config.MODEL_DIR / f'model_{fold}.pth', map_location='cpu', weights_only=True)
        assert np.isfinite(checkpoint['metric'])
        scores = np.load(config.SCORES_DIR / 'test' / f'scores_{fold}.npy')
        assert scores.shape == (3, 1, 101, 101) and np.isfinite(scores).all()
    folds = pd.read_csv(config.PROCESSED_DATA_DIR / 'train.csv')
    assert folds.id.is_unique and set(folds.fold) == {1, 2}
    print('DRY RUN OK: 8 train / 3 test images; 2 CPU folds; Lovasz training, checkpoint reload,')
    print('flip TTA, native-size OOF scoring, threshold search, fold ensemble, and RLE CSV. Skipped: none.')


if __name__ == '__main__':
    main()
