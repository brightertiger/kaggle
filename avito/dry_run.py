#!/usr/bin/env python3
"""Generate schema-compatible CSVs and exercise the compact pipeline on CPU."""
import os

# Set before numerical imports; no GPU libraries or pretrained weights are used.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import AvitoConfig, Config, ModelConfig
from src.data_utils import align_rows
from src.feature_engineering import TextPreprocessor, UserFeatures
from src.models import UserModel
from src.pipeline import AvitoPipeline

ROOT = Path(__file__).resolve().parent


def generate_sample(destination):
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2017)
    rows = []
    for i in range(64):
        # Non-sorted IDs detect accidental ordering by groupby/merge.
        rows.append({
            'item_id': f'item_{(i * 17) % 67:04d}', 'user_id': f'user_{i % 9}',
            'region': ['Москва', 'Свердловская область'][i % 2],
            'city': ['Москва', 'Екатеринбург', 'Нижний Тагил'][i % 3],
            'parent_category_name': 'Личные вещи',
            'category_name': ['Одежда', 'Детские товары', 'Аксессуары'][i % 3],
            'param_1': ['Новый', 'Бывший в употреблении'][i % 2],
            'param_2': None if i % 4 == 0 else 'Размер средний',
            'param_3': None,  # Deliberately constant/missing.
            'title': None if i % 7 == 0 else f'Продам куртку МОДНАЯ размер {i % 5}',
            'description': None if i % 6 == 0 else 'Хорошее состояние! Доставка по городу, звоните.',
            'price': None if i % 8 == 0 else float(100 + i * 25),
            'item_seq_number': i + 1,
            'activation_date': f'2017-03-{15 + i % 10:02d}',
            'user_type': ['Private', 'Company', 'Shop'][i % 3],
            'image': None,  # Missing images are valid competition records.
            'image_top_1': None if i % 5 == 0 else i % 3,
            'deal_probability': float(np.clip(0.1 + 0.15 * (i % 3) + rng.normal(0, 0.04), 0, 1)),
        })
    frame = pd.DataFrame(rows)
    train = frame.iloc[:36].copy()
    test = frame.iloc[36:48].drop(columns='deal_probability').copy()
    active = frame.iloc[48:].drop(columns=['deal_probability', 'image', 'image_top_1'])
    train.to_csv(destination / 'train.csv', index=False)
    test.to_csv(destination / 'test.csv', index=False)
    active.iloc[:8].to_csv(destination / 'train_active.csv', index=False)
    active.iloc[8:].to_csv(destination / 'test_active.csv', index=False)
    pd.DataFrame({'item_id': test.item_id, 'deal_probability': 0.0}).to_csv(
        destination / 'sample_submission.csv', index=False)
    return train, test


def main():
    data, output = ROOT / 'sample_data', ROOT / 'dry_run_output'
    train, test = generate_sample(data)
    config = Config(AvitoConfig(DATA_ROOT=str(data), OUTPUT_DIR=str(output),
                               FEATURES_DIR=str(output / 'features'),
                               MODEL_DIR=str(output / 'models'), N_FOLDS=3),
                    ModelConfig(TFIDF_MAX_FEATURES=128, RIDGE_MAX_ITER=100))
    pipeline = AvitoPipeline(config)
    pipeline.run_full_pipeline()
    diagnostics = pipeline.evaluate_pipeline()
    submission = pd.read_csv(output / 'submission.csv')
    assert list(submission.columns) == ['item_id', 'deal_probability']
    assert submission.item_id.tolist() == test.item_id.tolist()
    assert submission.deal_probability.between(0, 1).all()
    assert np.isfinite(submission.deal_probability).all()
    assert len(diagnostics['fold_scores']) == config.avito.N_FOLDS
    assert np.isfinite(diagnostics['mean_rmse'])

    # OOF predictions must cover every training row exactly once.
    for name, column in [('text', 'text_score'), ('user', 'user_score'),
                         ('ensemble', 'deal_probability')]:
        predictions = pd.read_csv(Path(config.avito.INSAMPLE_DIR) / f'{name}_model.csv')
        assert predictions.item_id.is_unique
        assert set(predictions.item_id) == set(train.item_id)
        assert np.isfinite(predictions[column]).all()
    oof = pd.read_csv(Path(config.avito.INSAMPLE_DIR) / 'ensemble_model.csv')
    paired = align_rows(train[['item_id', 'deal_probability']],
                        oof.rename(columns={'deal_probability': 'prediction'}), 'item_id')
    errors = (paired.deal_probability - paired.prediction) ** 2
    fold_errors = [errors[paired.item_id.isin(pipeline.data_loader.load_fold_data(f)[1].item_id)].mean() ** 0.5
                   for f in range(1, config.avito.N_FOLDS + 1)]
    np.testing.assert_allclose(diagnostics['fold_scores'], fold_errors)
    assert diagnostics['mean_rmse'] > 0  # Regression for the old self-comparison bug.

    # Feature row order must not affect fitted labels or submitted test IDs.
    model = UserModel(config)
    model.set_data_loader(pipeline.data_loader)
    expected_valid, expected_test = model.train_level1_model(1)
    user_path = Path(config.avito.FEATURES_DIR) / 'user/user_features.csv'
    features = pd.read_csv(user_path)
    features.sample(frac=1, random_state=2017).to_csv(user_path, index=False)
    actual_valid, actual_test = model.train_level1_model(1)
    pd.testing.assert_frame_equal(expected_valid, actual_valid)
    pd.testing.assert_frame_equal(expected_test, actual_test)
    features.to_csv(user_path, index=False)
    assert (UserFeatures(config)._normalize_feature(pd.Series([1.0, 1.0])) == 0).all()
    assert TextPreprocessor().clean_text('ПРИВЕТ, мир!') == 'привет мир'
    try:
        align_rows(test[['item_id']], submission.iloc[:-1], 'item_id')
    except ValueError:
        pass
    else:
        raise AssertionError('Missing predictions must be rejected')
    print(f'PASS: {len(train)} train / {len(test)} test rows; preprocess, features, '
          f'Ridge models, linear blend, evaluation, submission: {output / "submission.csv"}')
    print('No compact-pipeline stages skipped. Historical LightGBM/CNN/image/team ensemble '
          'not exercised; original external artifacts are absent.')


if __name__ == '__main__':
    main()
