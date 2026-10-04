"""Offline CPU smoke test; all generated files stay beside this script."""
from pathlib import Path
import hashlib
import os

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
from src.pipeline import JigsawPipeline
from src.utils.config import Config


class SmokeSentenceEncoder:
    """Deterministic shape-compatible stand-in for downloaded USE features."""
    def get_embeddings(self, texts):
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(str(text).encode()).digest()[:4], 'little')
            rows.append(np.random.default_rng(seed).normal(size=512))
        return np.asarray(rows, dtype=np.float32)


def write_sample(root):
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame({
        'id': [f'en_{i}' for i in range(40)],
        'comment_text': [f'{"hostile insult" if i % 2 else "helpful kind discussion"} example {i}' for i in range(40)],
        'toxic': [i % 2 for i in range(40)],
    })
    for column in ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        train[column] = 0
    train.to_csv(raw / 'jigsaw-toxic-comment-train.csv', index=False)
    bias = pd.DataFrame({
        'id': [f'bias_{i}' for i in range(10)],
        'comment_text': [f'English annotation example {i}' for i in range(10)],
        'toxic': [0.8 if i % 2 else 0.1 for i in range(10)],
        'severe_toxicity': [0.0] * 10,
    })
    bias.to_csv(raw / 'jigsaw-unintended-bias-train.csv', index=False)
    valid = pd.DataFrame({
        'id': list(range(100, 109)),
        'comment_text': [f'{"comentario hostil" if i % 2 else "grazie amico"} validation {i}' for i in range(9)],
        'lang': ['es', 'it', 'tr'] * 3, 'toxic': [i % 2 for i in range(9)],
    })
    valid.to_csv(raw / 'validation.csv', index=False)
    test = pd.DataFrame({'id': list(range(200, 207)),
                         'content': [f'bonjour discussion test {i}' for i in range(7)],
                         'lang': ['fr', 'es', 'it', 'tr', 'ru', 'pt', 'fr']})
    test.to_csv(raw / 'test.csv', index=False)
    pd.DataFrame({'id': test['id'], 'toxic': 0.0}).to_csv(raw / 'sample_submission.csv', index=False)
    # Optional labeled augmentation inputs exercise all domain-weighting branches.
    for filename, source in [('train_foreign.csv', 'translation'), ('subtitle.csv', 'subtitle')]:
        frame = valid.copy()
        frame['id'] = [f'{source}_{i}' for i in range(len(frame))]
        frame['comment_text'] = [f'{source} training sentence {i}' for i in range(len(frame))]
        frame.to_csv(raw / filename, index=False)


def main():
    folder = Path(__file__).resolve().parent
    data_dir, output_dir = folder / 'sample_data', folder / 'dry_run_output'
    write_sample(data_dir)
    torch.set_num_threads(1)
    config = Config(data_dir, output_dir, DEVICE='cpu', TINY=True, PRETRAINED=False,
                    MAX_LENGTH=32, BATCH_SIZE=3, N_FOLDS=2, EPOCHS_V1=1, EPOCHS_V2=1,
                    NUM_WORKERS=0, LGB_ROUNDS=3, LGB_EARLY_STOPPING=2)
    config.LGB_PARAMS.update(num_threads=1, num_leaves=4, min_child_weight=0,
                             min_data_in_leaf=1, feature_pre_filter=False)
    pipeline = JigsawPipeline(config, embedding_encoder=SmokeSentenceEncoder())
    predictions = pipeline.run_full_pipeline()
    train = pd.read_csv(data_dir / 'process/english/train_english.csv')
    valid = pd.read_csv(data_dir / 'process/english/valid_english.csv')
    assert set(train['comment_text']).isdisjoint(valid['comment_text'])
    pseudo = pd.read_csv(data_dir / 'process/pseudo/train_combine.csv')
    assert pseudo[['comment_text', 'toxic', 'weight']].notna().all().all()
    target = pd.read_csv(data_dir / 'raw/sample_submission.csv', dtype={'id': str})
    submission = pd.read_csv(output_dir / 'submission.csv', dtype={'id': str})
    assert list(submission.columns) == ['id', 'toxic']
    assert submission['id'].tolist() == target['id'].tolist()
    assert len(predictions) == len(target) and submission['id'].is_unique
    assert np.isfinite(submission['toxic']).all() and submission['toxic'].between(0, 1).all()
    for version in [1, 2]:
        for fold in range(config.N_FOLDS):
            assert (output_dir / f'version{version}/model_{fold}.pt').exists()
            assert len(pd.read_csv(output_dir / f'version{version}/score_{fold}.csv')) == len(target)
    for fold in range(config.N_FOLDS):
        first = torch.load(output_dir / f'version1/model_{fold}.pt', weights_only=True)
        second = torch.load(output_dir / f'version2/model_{fold}.pt', weights_only=True)
        assert not torch.equal(first['model_state_dict']['output.weight'],
                               second['model_state_dict']['output.weight'])
    first_scores = pd.read_csv(output_dir / 'version1/score_0.csv')['toxic']
    second_scores = pd.read_csv(output_dir / 'version1/score_1.csv')['toxic']
    assert not np.array_equal(first_scores, second_scores), 'Distinct checkpoints must be scored'
    print(f'PASS: CPU preprocessing, features, LightGBM weights, both XLM-R stages, '
          f'checkpoint inference and ensemble; {len(submission)} submission rows.')
    print('USE download/inference substituted with deterministic sentence features; pretrained weights not used.')
    print(f'Submission: {output_dir / "submission.csv"}')


if __name__ == '__main__':
    main()
