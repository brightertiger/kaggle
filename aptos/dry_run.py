#!/usr/bin/env python3
"""Exercise the real training code with synthetic fundus-like images, offline on CPU."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from main import main as run_cli
from src.config import Config
from src.data_utils import DiabeticRetinopathyDataset, ImageTransforms
from src.loss import DiabeticRetinopathyLoss, NoiseAugmentedLoss, VarianceLoss
from src.model import DiabeticRetinopathyModel
from src.pipeline import APTOSPipeline

ROOT = Path(__file__).resolve().parent
SAMPLE_ROOT = ROOT / 'sample_data'
OUTPUT_ROOT = ROOT / 'dry_run_output'


def make_sample_data():
    """Native APTOS PNG/CSV layout plus the labeled external JPEG dataset layout."""
    rng = np.random.default_rng(2017)

    def write_images(directory, prefix, extension, count):
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx in range(count):
            grade = idx % 5
            image_id = f'{prefix}_{idx:03d}'
            # Rectangular RGB photographs with a dark border and circular retinal field.
            height, width = 80, 112
            yy, xx = np.mgrid[:height, :width]
            mask = ((xx - width / 2) / 36) ** 2 + ((yy - height / 2) / 36) ** 2 < 1
            pixels = np.zeros((height, width, 3), dtype=np.uint8)
            color = np.array([120 + grade * 20, 65 + grade * 8, 35])
            texture = rng.integers(-15, 16, size=(height, width, 3))
            pixels[mask] = np.clip(color + texture[mask], 0, 255)
            Image.fromarray(pixels).save(directory / f'{image_id}{extension}')
            rows.append({'id_code': image_id, 'diagnosis': grade})
        return pd.DataFrame(rows)

    aptos = SAMPLE_ROOT / 'train'
    train = write_images(aptos / 'train_images', 'aptos', '.png', 10)
    train.to_csv(aptos / 'train.csv', index=False)
    test = write_images(aptos / 'test_images', 'test', '.png', 3)
    test[['id_code']].to_csv(aptos / 'test.csv', index=False)
    test.assign(diagnosis=0).to_csv(aptos / 'sample_submission.csv', index=False)
    external = SAMPLE_ROOT / 'pretrain'
    for split, filename in [('train', 'trainLabels15.csv'), ('test', 'testLabels15.csv')]:
        labels = write_images(external / split, f'2015_{split}', '.jpg', 10)
        labels.rename(columns={'id_code': 'image', 'diagnosis': 'level'}).to_csv(external / filename, index=False)
    return train, test


def check_invariants(config, train):
    dataset = DiabeticRetinopathyDataset(SAMPLE_ROOT / 'train' / 'train_images', train,
                                         config.IMAGE_SIZE, config=config)
    assert torch.equal(dataset[0]['image'], dataset[0]['image']), 'Validation must be deterministic'
    assert dataset[0]['image'].shape == (3, 64, 64)
    rectangular = Image.new('RGB', (112, 80), 'white')
    assert ImageTransforms(config)._resize(rectangular, 64).size == (64, 46)
    model = DiabeticRetinopathyModel(config.MODEL_NAME, config).eval()
    with torch.no_grad():
        regression, classification = model(dataset[0]['image'].unsqueeze(0))
    assert regression.shape == (1,) and classification.shape == (1, 5)
    assert DiabeticRetinopathyLoss()(regression, classification, torch.tensor([2.]), torch.ones(1)).shape == (1,)
    # Consistency penalties must be per image, not broadcast across the batch.
    assert torch.equal(VarianceLoss()(torch.tensor([1., 3.]), torch.zeros(2)), torch.tensor([1., 9.]))
    reg = torch.tensor([0.1, 1.9], requires_grad=True)
    cls = torch.zeros(2, 5, requires_grad=True)
    loss = NoiseAugmentedLoss()(reg, reg + 0.1, cls, cls + 0.1, torch.tensor([0., 2.]), torch.ones(2))
    assert loss.shape == (2,) and torch.isfinite(loss).all()
    loss.mean().backward()
    assert reg.grad is not None and cls.grad is not None


def main():
    torch.set_num_threads(2)
    train, test = make_sample_data()
    config = Config(DATA_ROOT=str(SAMPLE_ROOT), MODEL_SAVE_PATH=str(OUTPUT_ROOT),
                    MODEL_NAME='efficientnet-b0', PRETRAINED=False, DEVICE='cpu',
                    NUM_WORKERS=0, IMAGE_SIZE=64, LARGE_IMAGE_SIZE=64,
                    BATCH_SIZE=2, VALIDATION_BATCH_SIZE=2, USE_APEX=False,
                    PRETRAIN_FOLDS=2, TRAIN_FOLDS=2,
                    NUM_EPOCHS_PRETRAIN=1, NUM_EPOCHS_TRAIN=1, NUM_EPOCHS_COMBINE=1)
    # Fail the smoke test if any stage accidentally requests pretrained weights.
    with patch('src.model.EfficientNet.from_pretrained', side_effect=AssertionError('Dry run must stay offline')):
        check_invariants(config, train)
        common = ['--data-dir', str(SAMPLE_ROOT), '--model-dir', str(OUTPUT_ROOT),
                  '--model-name', 'efficientnet-b0', '--no-pretrained', '--device', 'cpu',
                  '--num-workers', '0', '--no-apex', '--image-size', '64', '--large-image-size', '64',
                  '--batch-size', '2', '--validation-batch-size', '2', '--pretrain-folds', '2',
                  '--train-folds', '2', '--epochs-pretrain', '1', '--epochs-train', '1',
                  '--epochs-combine', '1']
        run_cli(common + ['--step', 'all', '--fold', '1'])
        # Also run the optional two-view dataset/loss/trainer, in its own output directory.
        run_cli(common + ['--step', 'combine', '--fold', '1', '--noise-augmentation',
                          '--model-dir', str(OUTPUT_ROOT / 'noise')])
        checkpoints = [OUTPUT_ROOT / stage / 'model_1.pt' for stage in ('pretrain', 'train', 'combine')]
        checkpoints.append(OUTPUT_ROOT / 'noise' / 'combine' / 'model_1.pt')
        for path in checkpoints:
            saved = torch.load(path, map_location='cpu', weights_only=True)
            assert np.isfinite(saved['loss']) and saved['model_name'] == 'efficientnet-b0'
        submission_path = OUTPUT_ROOT / 'submission.csv'
        run_cli(common + ['--step', 'predict', '--checkpoints', *map(str, checkpoints),
                          '--submission', str(submission_path)])
        submission = pd.read_csv(submission_path)
        assert list(submission.columns) == ['id_code', 'diagnosis']
        assert submission['id_code'].tolist() == test['id_code'].tolist()
        assert submission['diagnosis'].isin(range(5)).all()
        assert pd.api.types.is_integer_dtype(submission['diagnosis'])
        # Keep this a wiring check: synthetic kappa says nothing about competition performance.
        print(f'DRY RUN PASS: preprocess → CNN features → pretrain → fine-tune → combine '
              f'→ paired-view training → checkpoint ensemble → {len(submission)} submission rows.')
        print(f'CPU, random EfficientNet-B0, no downloads. Output: {submission_path}')
        print('Skipped: CUDA/Apex acceleration and full-data competition training.')


if __name__ == '__main__':
    main()
