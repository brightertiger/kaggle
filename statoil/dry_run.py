"""Exercise every pipeline stage on synthetic radar data, using CPU and no downloads."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'dry_run_output' / 'matplotlib'))
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import Config
from src.pipeline import IcebergPipeline


def generate_sample(folder):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2017)
    yy, xx = np.mgrid[:75, :75]
    for split, count in [('train', 12), ('test', 4)]:
        records = []
        for index in range(count):
            label = index % 2
            blob = np.exp(-((xx - 37) ** 2 + (yy - 37) ** 2) / (60 + 100 * label))
            band_1 = (-20 + rng.normal(0, 2, (75, 75)) + 8 * blob).astype(np.float32)
            band_2 = (-25 + rng.normal(0, 2, (75, 75)) + (3 + 4 * label) * blob).astype(np.float32)
            if index == 0:  # Constant channels exercise zero-variance normalization.
                band_1.fill(-20)
                band_2.fill(-25)
            record = {'id': f'{split}_{index:04d}', 'band_1': band_1.ravel().tolist(),
                      'band_2': band_2.ravel().tolist(),
                      'inc_angle': 'na' if index % 4 == 0 else float(30 + index)}
            if split == 'train':
                record['is_iceberg'] = label
            records.append(record)
        pd.DataFrame(records).to_json(folder / f'{split}.json', orient='records')


def main():
    tf.config.set_visible_devices([], 'GPU')
    config = Config()
    config.DATA_DIR = str(ROOT / 'sample_data')
    config.MODEL_DIR = str(ROOT / 'dry_run_output' / 'models')
    config.SUBMISSION_DIR = str(ROOT / 'dry_run_output' / 'submissions')
    config.FOLDS = 2
    config.BATCH_SIZE = 2
    # 48 is large enough for the original valid-padding advanced CNN blocks.
    config.IMAGE_SIZE = 48
    config.CNN_WIDTH = 1 / 32
    config.VGG_WIDTH = 1 / 32
    config.VGG_WEIGHTS = None
    config.FEATURE_WORKERS = 1
    config.XGB_ROUNDS = 2
    config.XGB_EARLY_STOPPING = 1
    config.XGBOOST_PARAMS.update(nthread=1, max_depth=2, min_child_weight=1)
    for settings in config.MODEL_CONFIGS.values():
        settings.update(epochs=1, patience=1, steps_per_epoch=1)
        if 'fine_tune_epochs' in settings:
            settings['fine_tune_epochs'] = 1
    generate_sample(Path(config.DATA_DIR) / 'download')
    IcebergPipeline(config).run_full_pipeline()
    expected_ids = pd.read_json(Path(config.DATA_DIR) / 'download/test.json')['id']
    for name in ('cnn_basic', 'cnn_advanced', 'vgg16', 'ensemble', 'xgboost'):
        submission = pd.read_csv(Path(config.SUBMISSION_DIR) / f'{name}.csv')
        assert list(submission.columns) == ['id', 'is_iceberg']
        assert submission['id'].equals(expected_ids)
        assert np.isfinite(submission['is_iceberg']).all()
        assert submission['is_iceberg'].between(0, 1).all()
    for name in ('cnn_basic', 'cnn_advanced', 'vgg16'):
        oof = pd.read_csv(Path(config.DATA_DIR) / 'model' / f'{name}.csv')
        assert len(oof) == 12 and oof['id'].nunique() == 12
        assert np.isfinite(oof['score']).all()
    print('DRY RUN PASSED: preprocessing, features, both CNNs, frozen/fine-tuned VGG,')
    print('OOF prediction, confidence blend, XGBoost, and submission checks; CPU; no downloads; nothing skipped.')
    print(f'Outputs: {config.SUBMISSION_DIR}')


if __name__ == '__main__':
    main()
