"""Offline CPU smoke test of the production pipeline on synthetic GAP tables."""
import os
from pathlib import Path
import json

# Set before importing Transformers, including when invoked outside this folder.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import spacy
from spacy.training import Example
import torch
import yaml

from src.config import Config
from src.pipeline import PronounResolutionPipeline

ROOT = Path(__file__).resolve().parent


def make_rows(count, prefix):
    examples = [
        ('Alice met Beth after she finished work.', 'Alice', 'Beth', 'she', True, False),
        ('David thanked Charles because he helped.', 'David', 'Charles', 'he', False, True),
        ('Alice and Beth met Carol. Later she left.', 'Alice', 'Beth', 'she', False, False),
        ('Beth met Alice after she finished work.', 'Beth', 'Alice', 'she', True, False),
        ('Charles thanked David because he helped.', 'Charles', 'David', 'he', False, True),
        ('David and Charles met Edward. Later he left.', 'David', 'Charles', 'he', False, False),
    ]
    rows = []
    for i in range(count):
        text, a, b, pronoun, coref_a, coref_b = examples[i % len(examples)]
        rows.append({'ID': f'{prefix}-{i + 1}', 'Text': text, 'Pronoun': pronoun,
                     'Pronoun-offset': text.index(' ' + pronoun + ' ') + 1,
                     'A': a, 'A-offset': text.index(a), 'A-coref': coref_a,
                     'B': b, 'B-offset': text.index(b), 'B-coref': coref_b,
                     'URL': f'https://en.wikipedia.org/wiki/{a}'})
    return pd.DataFrame(rows)


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    spacy.util.fix_random_seed(42)
    sample = ROOT / 'sample_data'
    output = ROOT / 'dry_run_output'
    sample.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    train, val, test = make_rows(9, 'synthetic-train'), make_rows(5, 'synthetic-val'), make_rows(3, 'synthetic-test')
    train.to_csv(sample / 'gap-test.tsv', sep='\t', index=False)
    val.to_csv(sample / 'gap-validation.tsv', sep='\t', index=False)
    test.drop(columns=['A-coref', 'B-coref']).to_csv(sample / 'gap-development.tsv', sep='\t', index=False)

    tokenizer_dir = sample / 'tiny-bert'
    tokenizer_dir.mkdir(exist_ok=True)
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    words = sorted(set(' '.join(train['Text']).lower().replace('.', ' .').split()))
    (tokenizer_dir / 'vocab.txt').write_text('\n'.join(vocab + words) + '\n')
    (tokenizer_dir / 'tokenizer_config.json').write_text(json.dumps({'do_lower_case': True, 'tokenizer_class': 'BertTokenizer'}))

    # A locally initialized parser/NER executes the full feature code without
    # downloading en_core_web_lg. Its linguistic predictions are not meaningful.
    nlp = spacy.blank('en')
    parser = nlp.add_pipe('parser')
    for label in ('ROOT', 'nsubj', 'dobj', 'poss', 'compound', 'appos', 'pobj', 'xcomp', 'dep'):
        parser.add_label(label)
    ner = nlp.add_pipe('ner')
    ner.add_label('PERSON')
    doc = nlp.make_doc('Alice met Beth.')
    example = Example.from_dict(doc, {'heads': [1, 1, 1, 1],
                                      'deps': ['nsubj', 'ROOT', 'dobj', 'punct'],
                                      'entities': [(0, 5, 'PERSON'), (10, 14, 'PERSON')]})
    nlp.initialize(lambda: [example])
    nlp.to_disk(sample / 'tiny-spacy')

    values = {
        'model': {'pretrained_model': str(tokenizer_dir), 'random_init': True,
                  'hidden_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 4,
                  'intermediate_size': 64, 'freeze_layers': 0, 'max_length': 64,
                  'epochs': 1, 'batch_size': 3, 'dropout': 0.1},
        'data': {'train_path': str(sample / 'gap-test.tsv'),
                 'val_path': str(sample / 'gap-validation.tsv'),
                 'test_path': str(sample / 'gap-development.tsv'),
                 'spacy_model': str(sample / 'tiny-spacy'),
                 'output_dir': str(output), 'n_folds': 2, 'num_workers': 0},
        'device': 'cpu', 'seed': 42,
    }
    config_path = sample / 'dry_run_config.yaml'
    config_path.write_text(yaml.safe_dump(values))
    pipeline = PronounResolutionPipeline(Config.from_yaml(config_path))
    histories = pipeline.train()
    # Construct a fresh pipeline to exercise restoration as a separate predict run.
    submission = PronounResolutionPipeline(Config.from_yaml(config_path)).predict()
    saved = pd.read_csv(output / 'submission.csv')
    assert saved.columns.tolist() == ['ID', 'A', 'B', 'NEITHER']
    assert saved['ID'].tolist() == test['ID'].tolist()
    probabilities = submission[['A', 'B', 'NEITHER']].to_numpy()
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    for fold, history in histories.items():
        assert np.isfinite(history['train_loss'] + history['val_loss']).all()
        assert (output / f'fold_{fold}' / 'best_model.pth').is_file()
    print(f'DRY RUN PASS: {len(train) + len(val)} labeled rows, 2 folds, 1 epoch/fold, '
          f'{len(test)} predictions; CPU, offline, all pipeline stages ran.')
    print('BERT and spaCy are randomly initialized smoke-test models; no quality claim.')
    print(f'Submission: {output / "submission.csv"}')


if __name__ == '__main__':
    main()
