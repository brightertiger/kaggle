#!/usr/bin/env python3
"""Exercise every production stage on synthetic sentences, without GloVe downloads."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Set before any TensorFlow import; all caches and generated data stay in this folder.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'dry_run_output' / 'matplotlib')
os.environ['KERAS_HOME'] = str(ROOT / 'dry_run_output' / 'keras')

import nltk
import numpy as np
import pandas as pd

from src.config import Config
from src.pipeline import SpookyAuthorPipeline


def make_sample(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    # Invented phrases are fixtures, not quotations or evidence about the authors.
    phrases = {
        'EAP': 'The raven tapped upon my chamber while the midnight bell tolled',
        'HPL': 'Ancient cosmic shadows stirred beneath the nameless ruined city',
        'MWS': 'My creation awakened and I reflected upon the sorrow of humanity',
    }
    endings = ['in the rain.', 'as I watched.', 'beyond the window.', 'during the storm.',
               'without a sound.', 'until morning.', 'under the moon.', 'beside the door.',
               'in my dream.', 'with great fear.', 'while I waited.', 'at dusk.']
    rows = [{'id': f'id{index:05d}', 'text': f'{phrase} {ending}', 'author': author}
            for index, (author, phrase, ending) in enumerate(
                (author, phrase, ending) for ending in endings for author, phrase in phrases.items())]
    train = pd.DataFrame(rows)
    test = pd.DataFrame({'id': [f'id{index + 90000}' for index in range(6)],
                         'text': ['A raven waited inside the silent chamber.',
                                  'Cosmic ruins haunted the ancient city.',
                                  'My creation brought sorrow to humanity.',
                                  '', '!!!', 'I watched the shadows at midnight.']})
    train.to_csv(data_dir / 'train.csv', index=False)
    test.to_csv(data_dir / 'test.csv', index=False)
    submission = test[['id']].copy()
    for author in Config.AUTHOR_NAMES:
        submission[author] = 1 / Config.NUM_CLASSES
    submission.to_csv(data_dir / 'sample_submission.csv', index=False)
    return train, test


def main():
    data_dir, output_dir = ROOT / 'sample_data', ROOT / 'dry_run_output'
    train, test = make_sample(data_dir)
    config = Config()
    config.NLTK_DATA_DIR = output_dir / 'nltk_data'
    config.NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(config.NLTK_DATA_DIR))
    for package, resource in [('stopwords', 'corpora/stopwords'),
                              ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng')]:
        try:
            nltk.data.find(resource)
        except LookupError:
            if not nltk.download(package, download_dir=str(config.NLTK_DATA_DIR), quiet=True):
                raise RuntimeError(f'Could not download the small NLTK resource: {package}')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    config.N_FOLDS = 2
    config.NN_EPOCHS = 1
    config.NN_BATCH_SIZE = 8
    config.MAX_SEQUENCE_LENGTH = 24
    config.EMBEDDING_DIM = 8
    config.LSTM_UNITS = 8
    config.RANDOM_EMBEDDINGS = True
    config.SVD_COMPONENTS = 3
    config.XGB_NUM_ROUNDS = 3
    config.XGB_EARLY_STOPPING = 2
    config.XGB_PARAMS = {**config.XGB_PARAMS, 'nthread': 1, 'max_depth': 2, 'device': 'cpu'}
    pipeline = SpookyAuthorPipeline(data_dir, output_dir / 'models', output_dir / 'scores', config)
    cv_history, predictions = pipeline.run_full_pipeline()
    path = output_dir / 'scores' / config.XGB_SCORE
    saved = pd.read_csv(path)
    assert saved.columns.tolist() == ['id', *config.AUTHOR_NAMES]
    assert saved['id'].tolist() == test['id'].tolist()
    probabilities = saved[config.AUTHOR_NAMES].to_numpy()
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert len(cv_history) > 0 and np.isfinite(cv_history).all()
    pipeline.get_feature_importance()  # Also check checkpoint reloading.
    for filename in (config.TRAIN_NN_SCORE, config.TRAIN_LSTM_SCORE, config.TRAIN_NB_SCORE):
        scores = pd.read_csv(output_dir / 'scores' / filename)
        assert scores['id'].tolist() == train['id'].tolist()
        assert not scores.isna().any().any()
    print(f'DRY RUN PASS: {len(train)} training rows, {len(predictions)} predictions; '
          'text/POS, NB, SVD, pooled embeddings, LSTM, XGBoost, submission and model reload; skipped: none.')
    print(f'Submission: {path}')


if __name__ == '__main__':
    main()
