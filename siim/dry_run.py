#!/usr/bin/env python3
"""Exercise the actual training pipeline offline with synthetic RGB dermoscopy files."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.data_utils import MelanomaDataset
from src.inference import MelanomaInference, load_trained_model
from src.models import MelanomaClassifier
from src.pipeline import MelanomaPipeline

ROOT = Path(__file__).resolve().parent


def make_sample(data_dir):
    rng = np.random.default_rng(Config.SEED)
    # These are synthetic sample sizes, not competition dataset counts.
    for split, count in [('train', 24), ('test', 4)]:
        image_dir = data_dir / split
        image_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        diagnoses = ['unknown', 'melanoma', 'nevus', 'seborrheic keratosis']
        for index in range(count):
            image_name = f'ISIC_{split}_{index:07d}'
            diagnosis = diagnoses[index % len(diagnoses)]
            image = rng.integers(100, 240, size=(64, 64, 3), dtype=np.uint8)
            cv2.ellipse(image, (32, 32), (12 + index % 5, 10), index * 10, 0, 360,
                        (45, 65, 85), -1)
            if not cv2.imwrite(str(image_dir / f'{image_name}.jpg'), image):
                raise RuntimeError('Could not write synthetic image')
            row = dict(image_name=image_name, patient_id=f'IP_{split}_{index:05d}',
                       sex=['male', 'female', None][index % 3],
                       age_approx=[20, 45, 70, None][index % 4],
                       anatom_site_general_challenge=['torso', 'lower extremity',
                                                      'palms/soles', None][index % 4])
            if split == 'train':
                row.update(diagnosis=diagnosis,
                           benign_malignant='malignant' if diagnosis == 'melanoma' else 'benign',
                           target=int(diagnosis == 'melanoma'))
            rows.append(row)
        pd.DataFrame(rows).to_csv(data_dir / f'{split}.csv', index=False)
    test = pd.read_csv(data_dir / 'test.csv')
    pd.DataFrame({'image_name': test['image_name'], 'target': 0.0}).to_csv(
        data_dir / 'sample_submission.csv', index=False)


def main():
    torch.set_num_threads(2)
    data_dir = ROOT / 'sample_data'
    output_dir = ROOT / 'dry_run_output'
    make_sample(data_dir)
    config = Config(DEVICE='cpu', MODEL_NAME='efficientnet-b0', PRETRAINED=False,
                    IMAGE_SIZE=32, BATCH_SIZE=4, NUM_EPOCHS=1, N_FOLDS=2,
                    NUM_WORKERS=0, USE_APEX=False, CUTOUT_HOLES=2, CUTOUT_SIZE=8)
    pipeline = MelanomaPipeline(data_dir, output_dir / 'models', output_dir, config)
    scores, predictions = pipeline.run_full_pipeline(use_tta=True)
    submission = pd.read_csv(output_dir / 'submission.csv')
    assert list(submission) == ['image_name', 'target']
    assert submission['image_name'].tolist() == pipeline.test_metadata['image_name'].tolist()
    assert len(predictions) == len(pipeline.test_metadata)
    assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
    assert np.isfinite(scores).all() and np.isfinite(pipeline.oof_predictions).all()
    assert (output_dir / 'oof.csv').is_file()
    for fold in range(config.N_FOLDS):
        assert (output_dir / 'models' / f'melanoma_fold_{fold}.pt').is_file()
    # Verify deserialization reconstructs the small backbone without pretrained downloads.
    model = load_trained_model(output_dir / 'models' / 'melanoma_fold_0.pt',
                               MelanomaClassifier, device='cpu')
    dataset = MelanomaDataset(data_dir / 'test', pipeline.test_metadata,
                              is_training=False, config=config)
    reloaded = MelanomaInference(model, 'cpu').predict_single_fold(
        dataset, batch_size=config.BATCH_SIZE, num_workers=0, use_tta=False)
    original = MelanomaInference(pipeline.models[0], 'cpu').predict_single_fold(
        dataset, batch_size=config.BATCH_SIZE, num_workers=0, use_tta=False)
    np.testing.assert_allclose(reloaded, original, rtol=1e-6, atol=1e-7)
    print(f'DRY RUN OK: {len(pipeline.train_metadata)} train / {len(submission)} test images; '
          f'{config.N_FOLDS} folds, CPU random-init EfficientNet-B0, training, OOF, checkpoint '
          f'reload, TTA, averaging and submission. No core stages skipped. Output: {output_dir}')


if __name__ == '__main__':
    main()
