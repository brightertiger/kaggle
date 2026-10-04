"""Small offline checks for the failures repaired during migration."""
import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import Config, BERTClassifier, GPTClassifier, CustomLoss, ModelTrainer
from src.evaluation import BiasEvaluator, ModelEvaluator


class RegressionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = Config(device='cpu', random_init=True)
        self.config.bert_config['backbone_config'] = dict(
            vocab_size=32, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
        )
        self.config.gpt_config['backbone_config'] = dict(
            vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=32, pad_token_id=0, eos_token_id=1, bos_token_id=1,
        )

    def test_padding_does_not_change_scores(self):
        for cls in (BERTClassifier, GPTClassifier):
            model = cls(self.config).eval()
            with torch.no_grad():
                short = model(torch.tensor([[2, 4, 3]]), torch.tensor([[1, 1, 1]]))
                padded = model(torch.tensor([[2, 4, 3, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]]))
            if isinstance(short, tuple):
                short, padded = short[0], padded[0]
            torch.testing.assert_close(short, padded, atol=1e-6, rtol=1e-5)

    def test_single_row_loss(self):
        logits = torch.tensor([[0.2]], requires_grad=True)
        loss = CustomLoss.gpt_loss(logits, torch.tensor([0.8]), torch.tensor([2.0]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_soft_labels_and_threshold_boundary(self):
        evaluator = ModelEvaluator(self.config)
        result = evaluator.calculate_metrics(np.array([0.1, 0.5, 0.9]), np.array([0.1, 0.7, 0.9]))
        self.assertEqual(result['auc'], 1.0)
        bias = BiasEvaluator(self.config)
        frame = pd.DataFrame({'target': [0.1, 0.5], 'male': [0.5, 0.5], 'prediction': [0.1, 0.8]})
        self.assertEqual(bias.compute_subgroup_auc(frame, 'male'), 1.0)
        self.assertTrue(np.isnan(bias.compute_auc([1, 1], [0.2, 0.5])))
        self.assertEqual(bias.power_mean(pd.Series([0.0, 1.0]), -5), 0.0)

    def test_evaluation_rejects_wrong_fold(self):
        with self.assertRaisesRegex(ValueError, 'same IDs'):
            ModelEvaluator(self.config).evaluate_predictions(
                pd.DataFrame({'id': [1], 'prediction': [0.2]}),
                pd.DataFrame({'id': [2], 'target': [0.1]}),
            )

    def test_accumulation_flushes_remainder_and_validation_measures_loss(self):
        self.config.gpt_config['gradient_accumulation_steps'] = 2
        model = GPTClassifier(self.config)
        trainer = ModelTrainer(self.config, 'gpt')
        rows = [dict(input_ids=torch.tensor([2, 4, 3]), attention_mask=torch.ones(3, dtype=torch.long),
                     target=torch.tensor(0.8), weight=torch.tensor(1.), aux_labels=torch.zeros(6),
                     id=f'comment-{i}') for i in range(3)]
        loader = DataLoader(rows, batch_size=1)
        optimizer = trainer.setup_optimizer(model, num_training_steps=2)
        before = model.classifier.weight.detach().clone()
        loss = trainer.train_epoch(model, loader, optimizer, CustomLoss.gpt_loss, 'cpu')
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(trainer.scheduler.last_epoch, 2)
        self.assertFalse(torch.equal(before, model.classifier.weight))
        valid_loss, predictions = trainer.validate_model(model, loader, 'cpu')
        self.assertGreater(valid_loss, 0)
        self.assertEqual(predictions['id'].tolist(), [row['id'] for row in rows])


if __name__ == '__main__':
    unittest.main()
