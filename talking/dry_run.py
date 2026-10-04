#!/usr/bin/env python3
"""Exercise the complete CPU pipeline on deterministic, competition-shaped CSVs."""
from pathlib import Path

import numpy as np
import pandas as pd

from src import Config, TalkingDataPipeline
from src.data import FeatureEngineer
from src.models import ModelEnsemble, TalkingDataModel

ROOT = Path(__file__).resolve().parent


def generate_sample(raw_dir):
    """Keep real column names, timestamps, attribution labels, and separate ID spaces."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    def clicks(n_rows, day):
        app = rng.integers(1, 9, n_rows)
        hours = np.resize([4, 5, 9, 10, 13, 14], n_rows)
        frame = pd.DataFrame({
            'ip': rng.integers(1, 25, n_rows),
            'app': app,
            'device': rng.integers(1, 4, n_rows),
            'os': rng.integers(1, 7, n_rows),
            'channel': rng.integers(1, 9, n_rows),
            'click_time': (pd.Timestamp(day) + pd.to_timedelta(hours, unit='h')
                           + pd.to_timedelta(np.arange(n_rows), unit='s')),
        })
        return frame

    train = pd.concat([clicks(240, '2017-11-08'), clicks(240, '2017-11-09')],
                      ignore_index=True)
    train['is_attributed'] = ((train['app'] == 1) | (train.index % 17 == 0)).astype('uint8')
    # Real training CSVs also contain attributed_time; it must never be a predictor.
    train['attributed_time'] = (train['click_time'] + pd.Timedelta(minutes=1)).where(
        train['is_attributed'].eq(1))
    excluded = clicks(8, '2017-11-07').assign(
        click_time=pd.Timestamp('2017-11-07 00:00:00'),
        is_attributed=0, attributed_time=pd.NaT,
    )
    train = pd.concat([train, excluded], ignore_index=True)
    train.to_csv(raw_dir / 'train.csv', index=False)

    test = clicks(96, '2017-11-10')
    # Identical events with distinct IDs exercise occurrence-based matching.
    test.iloc[1] = test.iloc[0]
    test.insert(0, 'click_id', rng.permutation(np.arange(1000, 1096)))
    test.to_csv(raw_dir / 'test.csv', index=False)
    supplement = pd.concat([test.iloc[:80].drop(columns='click_id'),
                            clicks(48, '2017-11-10')], ignore_index=True)
    supplement.insert(0, 'click_id', np.arange(5000, 5000 + len(supplement)))
    supplement.to_csv(raw_dir / 'test_supplement.csv', index=False)
    pd.DataFrame({'click_id': test['click_id'], 'is_attributed': 0.0}).to_csv(
        raw_dir / 'sample_submission.csv', index=False)
    return test


def tiny_config():
    config = Config(ROOT / 'dry_run_output', raw_data_dir=ROOT / 'sample_data')
    config.NUM_BOOST_ROUND = 12
    config.EARLY_STOPPING_ROUNDS = 4
    config.LOG_EVALUATION_PERIOD = 0
    config.NUM_THREADS = 1
    for params in config.LGB_PARAMS.values():
        params.update(device_type='cpu', num_leaves=7, min_child_samples=5,
                      min_child_weight=0.001, scale_pos_weight=1.0,
                      colsample_bytree=1.0)
    return config


def main():
    config = tiny_config()
    test = generate_sample(config.RAW_DATA_DIR)
    pipeline = TalkingDataPipeline(config)
    submission = pipeline.run_full_pipeline()
    expected_columns = ['click_id', 'is_attributed']
    assert list(submission) == expected_columns
    np.testing.assert_array_equal(submission['click_id'], test['click_id'])
    assert np.isfinite(submission['is_attributed']).all()
    assert submission['is_attributed'].between(0, 1).all()
    saved = pd.read_csv(config.SUBMISSIONS_DIR / 'final_submission.csv')
    pd.testing.assert_frame_equal(saved, submission)

    train = pd.read_feather(config.MODELS_DIR / 'train_data.feather')
    valid = pd.read_feather(config.MODELS_DIR / 'valid_data.feather')
    features = pd.read_feather(config.MODELS_DIR / 'test_data.feather')
    assert len(train) == len(valid) == 240
    assert set(train.click_id).isdisjoint(valid.click_id)
    assert set(train.is_attributed) == set(valid.is_attributed) == {0, 1}
    feature_groups = (pipeline._load_count_features(), pipeline._load_unique_features(),
                      pipeline._load_ranking_features())
    engineered_names = [name for group in feature_groups for name in group]
    assert len(engineered_names) == 18
    assert set(engineered_names).issubset(features.columns)
    assert features[engineered_names].notna().all().all()
    mapping = pd.read_feather(config.PROCESSED_DATA_DIR / 'mapping.feather')
    assert len(mapping) == 80 and mapping.click_id.is_unique and mapping.old_id.is_unique
    assert pipeline._load_count_features()['ip_cnt'].ip_cnt.sum() == 240 + 128 + 16

    # New wrappers exercise model persistence instead of using trained objects.
    reloaded = TalkingDataPipeline(config)
    reloaded.generate_predictions()
    evaluations = reloaded.evaluate_models()
    assert all(np.isfinite(item['auc_score']) for item in evaluations.values())
    importance = reloaded.get_feature_importance()
    assert np.isfinite(importance.importance).all()
    model = TalkingDataModel(config, 'model_1')
    baseline = model.predict(features)
    assert np.ptp(baseline) > 0, 'Tiny training must learn actual splits'
    np.testing.assert_allclose(baseline, model.predict(features[features.columns[::-1]]))
    assert not {'click_id', 'day', 'click_time', 'attributed_time', 'is_attributed'} & set(
        model.model.feature_name())
    ensemble = ModelEnsemble(config)
    ensemble.load_models(config.MODEL_NAMES)
    np.testing.assert_allclose(ensemble.predict_ensemble(features), submission.is_attributed)
    validation_blend = ensemble.blend_predictions(
        [config.DATA_DIR / 'score' / f'{name}.csv' for name in config.MODEL_NAMES],
        config.DATA_DIR / 'score' / 'ensemble.csv', mode='score',
        model_names=config.MODEL_NAMES,
    )
    np.testing.assert_array_equal(validation_blend.is_attributed, valid.is_attributed)

    # A shuffled model CSV must align by ID; zero predictions must stay finite.
    source = config.DATA_DIR / 'submit' / 'model_2.csv'
    reversed_file = config.DATA_DIR / 'submit' / 'reversed.csv'
    pd.read_csv(source).iloc[::-1].to_csv(reversed_file, index=False)
    aligned = ensemble.blend_predictions(
        [config.DATA_DIR / 'submit' / 'model_1.csv', reversed_file],
        config.DATA_DIR / 'submit' / 'aligned.csv',
    )
    np.testing.assert_allclose(aligned.is_attributed, submission.is_attributed)
    np.testing.assert_array_equal(ensemble._normalize(np.zeros(3)), np.zeros(3))
    pd.read_csv(source).iloc[:-1].to_csv(reversed_file, index=False)
    try:
        ensemble.blend_predictions(
            [source, reversed_file], config.DATA_DIR / 'submit' / 'invalid.csv')
    except ValueError as error:
        assert 'Mismatched click IDs' in str(error)
    else:
        raise AssertionError('Mismatched prediction IDs were accepted')

    # The standalone next-click helper must use seconds with any datetime unit.
    pair = features.iloc[:2].copy()
    pair['click_time'] = pd.Series(pd.to_datetime([
        '2017-11-10 04:00:00', '2017-11-10 04:00:03'])).astype('datetime64[us]')
    next_click = FeatureEngineer(config).create_next_click_features(pair)
    np.testing.assert_array_equal(next_click.next_click, [3.0, -1.0])
    print(f'\nDRY RUN PASS: {len(train)} train / {len(valid)} validation / {len(test)} test; '
          f'{len(engineered_names)} engineered features; {config.NUM_MODELS} CPU models.')
    print(f'Submission: {config.SUBMISSIONS_DIR / "final_submission.csv"}')
    print('No stages skipped. Synthetic AUC is not a competition result.')
    return submission


if __name__ == '__main__':
    main()
