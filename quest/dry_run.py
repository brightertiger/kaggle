#!/usr/bin/env python3
"""Exercise the real CLI on synthetic QUEST data without network or GPU access."""
import os
from pathlib import Path
import subprocess
import sys

# Set these before importing Transformers, and inherit them in CLI subprocesses.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer

from src.config import Config
from src.data_utils import DataProcessor
from src.models import ModelFactory
from src.trainer import Trainer
from src.evaluator import Evaluator

ROOT = Path(__file__).resolve().parent


def create_sample(data_dir, output_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = data_dir / 'tokenizer'
    tokenizer_dir.mkdir(exist_ok=True)
    vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + (
        'how why what can i use a the this is to for and in with question answer '
        'please explain example clear useful detailed short compare code science '
        'math history music test body title solution works because result yes no '
        'one two three four five six seven eight nine zero . ? !'
    ).split()
    vocabulary = list(dict.fromkeys(vocabulary))
    vocab_path = tokenizer_dir / 'vocab.txt'
    vocab_path.write_text('\n'.join(vocabulary) + '\n', encoding='utf-8')
    tokenizer = BertTokenizer(str(vocab_path), do_lower_case=True)
    tokenizer.save_pretrained(tokenizer_dir)
    config = Config(
        pretrained=False, tokenizer_name=str(tokenizer_dir), local_files_only=True,
        device='cpu', use_apex=False, max_length=32, n_folds=2, num_epochs=1,
        batch_size=2, gradient_accumulation_steps=4, learning_rate=1e-3,
        data_dir=str(data_dir), model_dir=str(output_dir / 'models'),
        output_dir=str(output_dir),
        bert_config=dict(vocab_size=len(tokenizer), hidden_size=32,
                         num_hidden_layers=2, num_attention_heads=4,
                         intermediate_size=64, max_position_embeddings=32,
                         pad_token_id=tokenizer.pad_token_id),
    )
    topics = ['code', 'science', 'math', 'history', 'music']

    def row(index):
        topic = topics[(index // 2) % len(topics)]
        return dict(
            qa_id=index, question_title=f'How can I use {topic}?',
            question_body=f'Please explain this {topic} question with a clear example.',
            question_user_name=f'asker_{index // 2}',
            question_user_page=f'https://example.com/users/{index // 2}',
            answer=('This solution works because the result is useful.' if index % 2
                    else 'Yes! Use a short example to explain the answer.'),
            answer_user_name=f'answerer_{index}',
            answer_user_page=f'https://example.com/users/answerer_{index}',
            url=f'https://example.com/questions/{index // 2}',
            category='SCIENCE', host='example.com',
        )

    train = pd.DataFrame([row(i) for i in range(10)])
    # Include missing text and a long sequence to exercise both preprocessing paths.
    train.loc[0, 'question_title'] = None
    train.loc[1, 'answer'] = 'useful, example! ' * 220
    for j, label in enumerate(config.label_columns):
        train[label] = ((np.arange(len(train)) + j) % 5) / 4.0
    test = pd.DataFrame([row(i) for i in range(100, 103)])
    test.loc[0, 'answer'] = None
    train.to_csv(data_dir / 'train.csv', index=False)
    test.to_csv(data_dir / 'test.csv', index=False)
    submission = pd.DataFrame(0.0, index=range(len(test)), columns=config.label_columns)
    submission.insert(0, 'qa_id', test.qa_id)
    submission.to_csv(data_dir / 'sample_submission.csv', index=False)
    config.save_config(output_dir / 'config.json')
    return config, train, test


def main():
    torch.set_num_threads(1)
    data_dir, output_dir = ROOT / 'sample_data', ROOT / 'dry_run_output'
    config, train, test = create_sample(data_dir, output_dir)
    config_path = output_dir / 'config.json'
    for mode in ('train', 'evaluate', 'inference'):
        subprocess.run([sys.executable, str(ROOT / 'main.py'), '--mode', mode,
                        '--config', str(config_path)], cwd=ROOT, check=True)

    result = pd.read_csv(output_dir / 'submission.csv')
    assert list(result.columns) == ['qa_id'] + config.label_columns
    assert result.qa_id.tolist() == test.qa_id.tolist()
    predictions = result[config.label_columns].to_numpy()
    assert predictions.shape == (len(test), config.num_labels)
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 1)).all()
    oof = pd.read_csv(Path(config.model_dir) / 'oof_predictions' / 'predictions.csv')
    assert oof.qa_id.tolist() == train.qa_id.tolist()
    for fold in range(1, config.n_folds + 1):
        train_fold = pd.read_csv(data_dir / 'split' / f'text_train_{fold}.csv')
        valid_fold = pd.read_csv(data_dir / 'split' / f'text_valid_{fold}.csv')
        assert set(train_fold.question_body).isdisjoint(valid_fold.question_body)
        checkpoint = torch.load(Path(config.model_dir) / f'fold_{fold}' / 'best_model.pt',
                                map_location='cpu', weights_only=True)
        # These loaders have fewer batches than accumulation_steps: a real update
        # proves the final partial accumulation window was not discarded.
        assert all(int(state['step']) > 0 for state in checkpoint['optimizer_state_dict']['state'].values())
        assert checkpoint['optimizer_state_dict']['state']

    # Preserve and verify the separate-encoder alternative and checkpoint resume.
    config.model_type = 'dual_bert'
    trainer = Trainer(config)
    before = trainer.model.classifier.weight.detach().clone()
    trainer.train_fold(1, str(data_dir), str(output_dir / 'dual_models'))
    assert not torch.equal(before, trainer.model.classifier.weight)
    checkpoint_path = str(output_dir / 'dual_models' / 'fold_1' / 'best_model.pt')
    trainer.load_model(checkpoint_path)
    reloaded = Evaluator(config).load_model(checkpoint_path)
    _, loader = DataProcessor(config).create_data_loaders(1, str(data_dir), 2)
    batch = next(iter(loader))
    trainer.model.eval()
    with torch.no_grad():
        expected = trainer.model(batch['question'], batch['answer'])
        actual = reloaded(batch['question'], batch['answer'])
    torch.testing.assert_close(actual, expected)
    ensemble = ModelFactory.create_ensemble(
        [{'type': 'dual_bert', 'weight': 2}, {'type': 'dual_bert', 'weight': 1}], config
    )
    assert torch.isclose(ensemble.weights.sum(), torch.tensor(1.0))
    print(f'\nPASS: {len(train)} training rows, {len(test)} test rows, '
          f'{config.num_labels} targets; grouped CV, training, checkpoint reload, '
          'OOF evaluation, fold averaging and submission passed on CPU.')
    print(f'Submission: {output_dir / "submission.csv"}')
    print('No pipeline stages skipped; pretrained downloads and CUDA/Apex are disabled.')


if __name__ == '__main__':
    main()
