#!/usr/bin/env python3
"""Offline CPU smoke run of both model families and every pipeline stage."""
import json
import os
from pathlib import Path

# Enforce offline mode before importing Hugging Face libraries.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import BertTokenizer, GPT2Tokenizer

from src import Config, JigsawPipeline


ROOT = Path(__file__).resolve().parent


def create_sample(config):
    data_path = Path(config.data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    rows = []
    # Every weight stratum has both targets and every identity has both classes.
    for repeat in range(12):
        for group in range(4):
            toxic = group % 2
            mentioned = group // 2
            row = {
                'id': 1000 + len(rows),
                'comment_text': ('This is an awful insulting comment.' if toxic else 'Thank you for a thoughtful kind comment.')
                                + (' A community member wrote it.' if mentioned else ' We can discuss this.'),
                'target': 0.8 if toxic else 0.1,
            }
            row.update({col: (0.7 if toxic else 0.0) for col in config.aux_labels if col != 'target'})
            row.update({col: float(mentioned) for col in config.identity_columns})
            rows.append(row)
    train = pd.DataFrame(rows)
    train.loc[0, 'comment_text'] = None
    train.loc[1, config.identity_columns] = np.nan
    train.loc[2, 'obscene'] = np.nan
    train.loc[3, 'target'] = 0.5
    test = pd.DataFrame({'id': range(2000, 2005), 'comment_text': [
        'Thank you for your help.', 'That was an awful comment.', None,
        'A community member has a different opinion.', 'We can discuss this calmly.',
    ]})
    train.to_csv(data_path / 'train.csv', index=False)
    test.to_csv(data_path / 'test.csv', index=False)
    pd.DataFrame({'id': test['id'], 'prediction': 0.0}).to_csv(data_path / 'sample_submission.csv', index=False)
    return train, test


def create_tokenizers(config, texts):
    token_dir = Path(config.data_path) / 'tokenizers'
    bert_dir, gpt_dir = token_dir / 'bert', token_dir / 'gpt'
    bert_dir.mkdir(parents=True, exist_ok=True)
    gpt_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    vocabulary += sorted(set(' '.join(texts).lower().replace('.', ' .').split()) - set(vocabulary))
    vocab_path = bert_dir / 'vocab.txt'
    vocab_path.write_text('\n'.join(vocabulary) + '\n')
    # Passing an explicit vocab works on both Transformers 4 and 5.
    bert = BertTokenizer(vocab=str(vocab_path), vocab_file=str(vocab_path), do_lower_case=True)
    bert.save_pretrained(bert_dir)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(texts, vocab_size=300, min_frequency=1, special_tokens=['<|endoftext|>'])
    bpe.save_model(str(gpt_dir))
    gpt = GPT2Tokenizer(vocab_file=str(gpt_dir / 'vocab.json'), merges_file=str(gpt_dir / 'merges.txt'),
                        vocab=json.loads((gpt_dir / 'vocab.json').read_text()),
                        merges=[tuple(line.split()) for line in (gpt_dir / 'merges.txt').read_text().splitlines()[1:] if line.strip()])
    gpt.pad_token = gpt.eos_token
    gpt.save_pretrained(gpt_dir)
    config.bert_config.update(tokenizer_name=str(bert_dir), backbone_config={
        'vocab_size': len(bert), 'hidden_size': 32, 'num_hidden_layers': 2,
        'num_attention_heads': 4, 'intermediate_size': 64,
        'max_position_embeddings': 64, 'pad_token_id': bert.pad_token_id,
    })
    config.gpt_config.update(tokenizer_name=str(gpt_dir), backbone_config={
        'vocab_size': len(gpt), 'n_embd': 32, 'n_layer': 2, 'n_head': 4, 'n_positions': 64,
        'pad_token_id': gpt.pad_token_id, 'eos_token_id': gpt.eos_token_id, 'bos_token_id': gpt.bos_token_id,
    })


def main():
    torch.set_num_threads(1)
    config = Config(data_path=str(ROOT / 'sample_data'), model_path=str(ROOT / 'dry_run_output' / 'models'),
                    output_path=str(ROOT / 'dry_run_output'), device='cpu', random_init=True, n_folds=2, max_length=32)
    config.training_config.update(num_workers=0, pin_memory=False, drop_last=False)
    for settings in (config.bert_config, config.gpt_config):
        settings.update(num_epochs=1, batch_size=5, valid_batch_size=5, learning_rate=1e-3,
                        gradient_accumulation_steps=2)
    train, test = create_sample(config)
    create_tokenizers(config, train['comment_text'].fillna('none blank').tolist())
    config.create_directories()
    config_path = ROOT / 'dry_run_output' / 'config.json'
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    result = JigsawPipeline(config).run_full_pipeline('both')
    for name, predictions in result['test_predictions'].items():
        assert predictions.columns.tolist() == ['id', 'prediction'], name
        assert predictions['id'].tolist() == test['id'].tolist(), name
        assert np.isfinite(predictions['prediction']).all(), name
        assert predictions['prediction'].between(0, 1).all(), name
    for fold in result['models'].values():
        for model in fold.values():
            assert np.isfinite(model['best_loss']) and model['best_loss'] > 0
    for fold in result['evaluations'].values():
        for metrics in fold.values():
            assert np.isfinite(metrics['final_metric'])
    # A new process-style pipeline must recover saved evaluation artifacts.
    fresh = JigsawPipeline(config)
    assert fresh.evaluate_models().keys() == result['evaluations'].keys()
    print(f'DRY RUN PASSED: {len(train)} train rows, {len(test)} test rows; BERT + GPT-2, '
          f'{config.n_folds} folds, CPU, local tokenizers, submission ensemble. No stages skipped.')
    print(f'Artifacts: {config.output_path}')


if __name__ == '__main__':
    main()
