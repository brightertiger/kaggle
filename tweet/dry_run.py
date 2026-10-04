#!/usr/bin/env python3
"""Offline CPU smoke test using the same CLI, model, and loaders as real data."""
import os
from pathlib import Path
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import torch
from tokenizers import ByteLevelBPETokenizer
from src.config import Config, DataConfig, ModelConfig, TrainingConfig, get_config
from src.data_utils import TweetDataset, normalize_text
from src.evaluator import TweetEvaluator
from src.models import TweetSentimentModel, TweetLoss


def create_sample():
    sample = ROOT / 'sample_data'
    output = ROOT / 'dry_run_output'
    tokenizer_dir = sample / 'tokenizer'
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    examples = [
        ('I love this sunny day!', 'love this sunny day', 'positive'),
        ('Such a wonderful morning', 'wonderful', 'positive'),
        ('This café is great!', 'great!', 'positive'),
        ('I am  really HAPPY today', 'really HAPPY', 'positive'),
        ('I hate this cold rain', 'hate this cold rain', 'negative'),
        ('This is a terrible delay', 'terrible', 'negative'),
        ('Feeling sad about the news', 'sad', 'negative'),
        ('The service was awful!', 'awful!', 'negative'),
        ('The bus arrives at noon', 'The bus arrives at noon', 'neutral'),
        ('I went to the shop', 'I went to the shop', 'neutral'),
        ('We are meeting today', 'We are meeting today', 'neutral'),
        ('There is a book on my desk', 'There is a book on my desk', 'neutral'),
    ]
    train = pd.DataFrame(examples, columns=['text', 'selected_text', 'sentiment'])
    train.insert(0, 'textID', [f'train{i:05d}' for i in range(len(train))])
    test = pd.DataFrame([
        ('0000000001', 'I love this café!', 'positive'),
        ('0000000002', 'The delay was awful!', 'negative'),
        ('0000000003', 'The bus arrives today', 'neutral'),
        ('0000000004', '', 'neutral'),
        ('0000000005', 'This is great! ' * 40, 'positive'),
    ], columns=['textID', 'text', 'sentiment'])
    train.to_csv(sample / 'train.csv', index=False)
    test.to_csv(sample / 'test.csv', index=False)
    pd.DataFrame({'textID': test.textID, 'selected_text': ''}).to_csv(sample / 'sample_submission.csv', index=False)
    tokenizer = ByteLevelBPETokenizer(lowercase=True, add_prefix_space=True)
    tokenizer.train_from_iterator(
        list(train.text) + list(train.sentiment), vocab_size=280, min_frequency=2,
        special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'],
    )
    tokenizer.save_model(str(tokenizer_dir))
    config = Config(
        data=DataConfig(train_path=str(sample / 'train.csv'), test_path=str(sample / 'test.csv'),
                        processed_path=str(output / 'processed'), model_path=str(output / 'models'),
                        vocab_file=str(tokenizer_dir / 'vocab.json'), merges_file=str(tokenizer_dir / 'merges.txt'),
                        max_length=96, batch_size=4, num_workers=0, n_folds=2),
        model=ModelConfig(pretrained=False, hidden_size=32, num_hidden_layers=4,
                          num_attention_heads=4, intermediate_size=64, vocab_size=tokenizer.get_vocab_size(),
                          max_epochs=1, gradient_accumulation_steps=3),
        training=TrainingConfig(device='cpu'),
    )
    config_path = output / 'config.json'
    config_path.write_text(json.dumps(config.to_dict(), indent=2) + '\n')
    return config, config_path


def check_alignment(config):
    """Regression checks for offset normalization, labels, truncation, and label-free input."""
    config.data.train_path = str(ROOT / 'dry_run_output' / 'processed' / 'train.csv')
    evaluator = TweetEvaluator(config)
    for fold in range(config.data.n_folds):
        dataset = TweetDataset(config.data.train_path, -fold - 1, config)
        for index, row in dataset.data.iterrows():
            item = dataset[index]
            assert item['tokens'].shape == item['aux_label'].shape == (config.data.max_length,)
            decoded = evaluator.extract_selected_text(row.text, item['start_idx'], item['end_idx'],
                                                       evaluator.get_offsets(row.text, row.sentiment))
            assert decoded == normalize_text(row.selected_text).strip(), (decoded, row.selected_text)
            assert item['text_mask'][item['start_idx']] and item['text_mask'][item['end_idx']]
    # Labels beyond the encoded window must fail explicitly, not become token zero.
    dataset.data.loc[0, 'text'] = 'word ' * 200 + 'answer'
    dataset.data.loc[0, 'selected_text'] = 'answer'
    try:
        dataset[0]
    except ValueError as error:
        assert 'truncated' in str(error)
    else:
        raise AssertionError('Truncated training target was accepted')
    # Exercise the formerly mismatched auxiliary tensors even though CE is the default.
    config.model.auxiliary_loss_weight = 1.0
    loss = TweetLoss(config)(torch.zeros(1, 96), torch.zeros(1, 96),
                             torch.tensor([4]), torch.tensor([5]), torch.zeros(1, 96),
                             torch.zeros(1, 96))
    assert torch.isfinite(loss)
    config.model.auxiliary_loss_weight = 0.0
    test = TweetDataset(config.data.test_path, 0, config, is_training=False)
    assert len(test) == 5 and 'start_idx' not in test[0]
    assert all(test[i]['tokens'].shape == (96,) for i in range(len(test)))
    assert len(test.tokenizer.encode('positive').ids) > 1  # dynamic sentiment prefix
    assert evaluator.jaccard_score('', '') == 1.0
    assert evaluator.jaccard_score('love this', 'love') == 0.5
    assert evaluator.extract_selected_text('  I LOVE  this ', 8, 2, []) == 'i love this'


def main():
    torch.set_num_threads(1)
    config, path = create_sample()
    for mode in ['train', 'predict']:
        subprocess.run([sys.executable, str(ROOT / 'main.py'), '--mode', mode, '--config', str(path),
                        '--output-path', str(ROOT / 'dry_run_output')], cwd=ROOT, check=True)
    check_alignment(config)
    submission = pd.read_csv(ROOT / 'dry_run_output' / 'submission.csv', dtype={'textID': str}, keep_default_na=False)
    test = pd.read_csv(config.data.test_path, dtype={'textID': str}, keep_default_na=False)
    assert list(submission.columns) == ['textID', 'selected_text']
    assert submission.textID.tolist() == test.textID.tolist()
    for text, selected in zip(test.text, submission.selected_text):
        assert selected in normalize_text(text), (selected, text)
        assert selected or not text.strip()
    # With fewer batches than accumulation_steps, weights must still have updated.
    torch.manual_seed(config.data.random_seed)
    initial = TweetSentimentModel(config)
    saved = torch.load(Path(config.data.model_path) / 'model_fold_0.pt', weights_only=True)
    assert not torch.equal(initial.classifier.weight, saved['model_state_dict']['classifier.weight'])
    for fold in range(config.data.n_folds):
        checkpoint = torch.load(Path(config.data.model_path) / f'model_fold_{fold}.pt', weights_only=True)
        assert torch.isfinite(torch.tensor(checkpoint['loss']))
    assert get_config(path).training.device == 'cpu'
    print('Dry run passed: 12 training rows, 2 folds, 1 epoch/fold, 5 submission rows.')
    print('Preprocessing → BPE features → RoBERTa training → checkpoint reload → Jaccard → ensemble → CSV.')
    print('CPU only; no downloads; no pipeline stages skipped. Output: dry_run_output/submission.csv')


if __name__ == '__main__':
    main()
