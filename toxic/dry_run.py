#!/usr/bin/env python3
"""Offline CPU smoke test of all model families using competition-shaped CSVs."""

import json
import os
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'dry_run_output' / 'matplotlib')

import numpy as np
import pandas as pd

from src.config import get_config
from src.data_utils import read_csv
from src.pipeline import ToxicCommentPipeline


def create_sample(data_dir: Path):
    """Use independent binary targets, varied text, and original competition headers."""
    data_dir.mkdir(parents=True, exist_ok=True)
    targets = get_config().evaluation.target_columns
    # Every label combination ensures both classes in these deterministic folds.
    labels = ((np.arange(64)[:, None] >> np.arange(len(targets))) & 1).astype(int)
    cues = ['rude', 'vicious', 'filthy', 'menacing', 'insulting', 'hateful']
    comments = [
        'Discussion edit ' + ' '.join(word if value else 'helpful' for word, value in zip(cues, row))
        + (' WOW!!! #TalkPage :-) https://example.org 127.0.0.1' if i % 3 == 0 else ' thank you')
        for i, row in enumerate(labels)
    ]
    comments[0] = ''
    train = pd.DataFrame(labels, columns=targets)
    train.insert(0, 'comment_text', comments)
    train.insert(0, 'id', [f'{i:016x}' for i in range(len(train))])
    test = pd.DataFrame({
        'id': [f'{i:016x}' for i in [109, 101, 108, 102, 107, 103, 106, 104]],
        'comment_text': ['', 'Helpful discussion!', 'RUDE rude rude!!!', 'A vicious edit',
                         'Thanks @editor :-)', 'Filthy insulting words', '#TalkPage good work',
                         'A hateful and menacing comment'],
    })
    train.to_csv(data_dir / 'train.csv', index=False)
    test.to_csv(data_dir / 'test.csv', index=False)
    sample_submission = test[['id']].copy()
    sample_submission[targets] = 0.0
    sample_submission.to_csv(data_dir / 'sample_submission.csv', index=False)
    return train, test


def main():
    data_dir, output_dir = ROOT / 'sample_data', ROOT / 'dry_run_output'
    train, test = create_sample(data_dir)
    config = get_config()
    config.data.train_path = str(data_dir / 'train.csv')
    config.data.test_path = str(data_dir / 'test.csv')
    config.data.output_dir = str(output_dir / 'processed')
    config.data.n_folds = 2
    # Exercise every preprocessing variant; neural training uses the first two.
    config.model.model_dir = str(output_dir / 'validation')
    config.model.log_dir = str(output_dir / 'logs')
    config.model.submission_dir = str(output_dir / 'submissions')
    config.model.seq_length = 16
    config.model.embed_size = 8
    config.model.usable_vocab = 128
    config.model.rnn_units = 4
    config.model.dense_units = 8
    config.model.batch_size = 16
    config.model.epochs = 1
    config.model.patience = 1
    config.model.recurrent_dropout = 0.0
    config.model.random_embeddings = True
    config.model.cpu_only = True
    config.model.nb_min_df = 1
    config.model.word_max_features = 128
    config.model.char_max_features = 256
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / 'config.json'
    config_path.write_text(json.dumps(asdict(config), indent=2) + '\n')
    # Round-trip the same JSON configuration accepted by main.py.
    pipeline = ToxicCommentPipeline(get_config(str(config_path)))
    predictions, scores = pipeline.run_full_pipeline()
    targets = config.evaluation.target_columns
    for name, frame in predictions.items():
        expected = test if name.endswith('_test') else train
        assert frame['id'].tolist() == expected['id'].tolist(), name
        assert frame.columns.tolist() == ['id', *targets], name
        values = frame[targets].to_numpy()
        assert np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all(), name
    assert all(np.isfinite(metrics['overall']) for metrics in scores.values())
    submission_path = Path(config.model.submission_dir, 'final_ensemble_submission.csv')
    submission = read_csv(submission_path)
    assert submission['id'].tolist() == test['id'].tolist()
    np.testing.assert_allclose(submission[targets], predictions['final_ensemble_test'][targets])
    print(f'PASS: {len(train)} training rows, {len(test)} test rows; preprocessing, GRU, NB-SVM, '
          'TF-IDF logistic regression, OOF evaluation, blending, and submission all ran on CPU.')
    print(f'No downloads. No stages skipped. Submission: {submission_path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
