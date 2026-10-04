"""Exercise the real pipeline on synthetic RGB JPEGs, on CPU without downloads."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import torch
from src.pipeline import CassavaPipeline
from src.utils.config import Config
from src.models.loss import CELoss, FocalLoss, LabelSmoothingLoss
from src.training.scoring import generate_classification_report


ROOT = Path(__file__).resolve().parent


def generate_sample(data_dir):
    rng = np.random.default_rng(42)
    for split in ('train_images', 'test_images'):
        (data_dir / split).mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(20):
        label = index % 5
        image_id = f'{100000 + index}.jpg'
        # JPEG layout matches the competition; patterns carry no disease meaning.
        image = rng.integers(0, 80, size=(64, 96, 3), dtype=np.uint8)
        cv2.ellipse(image, (48, 32), (30, 20), label * 20, 0, 360,
                    (30 + label * 20, 120, 40), thickness=-1)
        cv2.circle(image, (30 + label * 6, 32), 5, (150, 80, 80), thickness=-1)
        if not cv2.imwrite(str(data_dir / 'train_images' / image_id), image):
            raise RuntimeError('Failed to write synthetic JPEG')
        records.append(dict(image_id=image_id, label=label))
    pd.DataFrame(records).to_csv(data_dir / 'train.csv', index=False)
    test_ids = []
    for index in range(4):
        image_id = f'{200000 + index}.jpg'
        image = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
        if not cv2.imwrite(str(data_dir / 'test_images' / image_id), image):
            raise RuntimeError('Failed to write synthetic JPEG')
        test_ids.append(image_id)
    pd.DataFrame({'image_id': test_ids, 'label': 0}).to_csv(data_dir / 'sample_submission.csv', index=False)
    (data_dir / 'label_num_to_disease_map.json').write_text(
        json.dumps(dict(enumerate(Config.CLASS_NAMES)), indent=2) + '\n')
    return test_ids


def main():
    torch.set_num_threads(2)
    data_dir, output_dir = ROOT / 'sample_data', ROOT / 'dry_run_output'
    test_ids = generate_sample(data_dir)
    config = Config(DATA_DIR=str(data_dir), OUTPUT_DIR=str(output_dir),
                    MODEL_NAME='efficientnet_b0', PRETRAINED=False, DEVICE='cpu',
                    IMAGE_SIZE=64, BATCH_SIZE=3, NUM_WORKERS=0, EPOCHS=1,
                    N_FOLDS=2, SWA_START=0, ACCUMULATION_STEPS=3, NUM_TTA=2,
                    VERSIONS=('version0', 'version7'))
    pipeline = CassavaPipeline(config)
    results = pipeline.run_full_pipeline()
    submission = pd.read_csv(output_dir / 'submission.csv')
    assert list(submission.columns) == ['image_id', 'label']
    assert submission['image_id'].tolist() == test_ids
    assert submission['label'].between(0, 4).all()
    assert len(results) == config.N_FOLDS
    for version in config.VERSIONS:
        for split, count in [('oof', 20), ('test', 4)]:
            frame = pd.read_csv(output_dir / 'scores' / f'{version}_{split}.csv')
            probabilities = frame[[f'prob_{i}' for i in range(5)]].to_numpy()
            assert len(frame) == count and frame['image_id'].is_unique
            assert np.isfinite(probabilities).all()
            assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        for fold in range(config.N_FOLDS):
            checkpoint = output_dir / 'models' / version / f'model_{fold}_swa.pt'
            assert checkpoint.is_file(), 'SWA must execute in the smoke test'
            model, _ = pipeline._load_model(version, fold)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(state['model_state_dict'])
    # Guard the singleton-label regression in every supported loss and reporting.
    for loss in (CELoss(), FocalLoss(), LabelSmoothingLoss()):
        logits = torch.randn(1, 5, requires_grad=True)
        value = loss(logits, torch.tensor([0]))
        assert torch.isfinite(value)
        value.backward()
    generate_classification_report(np.array([[1, 0, 0, 0, 0]]), [0], Config.CLASS_NAMES)
    print('Dry run passed: 20 train / 4 test images; CPU random-init EfficientNet-B0; '
          'preprocessing, CNN features, training, SWA, OOF, TTA, blending and submission. '
          'No stages skipped; no pretrained downloads.')
    print(f'Submission: {output_dir / "submission.csv"}')


if __name__ == '__main__':
    main()
