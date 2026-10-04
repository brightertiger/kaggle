"""Regression checks for the repaired segmentation boundaries."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

from src import Config
from src.data.data_utils import crop_padding, restore_logits
from src.inference import ModelEvaluator, ModelPredictor
from src.models import create_loss_function, create_model, IOUMetric
from src.training import ModelTrainer


TEST_OUTPUT = Path(__file__).resolve().parents[1] / 'dry_run_output'


class PipelineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        TEST_OUTPUT.mkdir(exist_ok=True)

    def test_column_major_rle_and_empty_mask(self):
        mask = np.array([[1, 0, 1], [1, 0, 0]], dtype=np.uint8)
        self.assertEqual(ModelPredictor._rle_encode(mask), '1 2 5 1')
        self.assertEqual(ModelPredictor._rle_encode(np.zeros((2, 3))), '')
        self.assertEqual(ModelPredictor._rle_encode(np.ones((2, 3))), '1 6')

    def test_iou_empty_and_strict_thresholds(self):
        metric = IOUMetric(cutoff=0.5)
        empty = torch.zeros(1, 1, 2, 5)
        full = torch.ones_like(empty)
        self.assertEqual(metric(empty, empty), 1)
        self.assertEqual(metric(empty, full), 0)
        self.assertEqual(metric(full, empty), 0)
        self.assertEqual(metric(full, full), 1)
        six_pixels = torch.tensor([1.] * 6 + [0.] * 4).reshape_as(full)
        self.assertAlmostEqual(metric(six_pixels, full), 0.2)
        self.assertEqual(IOUMetric(cutoff=0.5, min_salt_pixels=6)(six_pixels, empty), 1)

    def test_padding_is_removed_on_spatial_axes(self):
        config = Config()
        values = torch.full((2, 1, 128, 128), -10.)
        values[..., 14:115, 14:115] = 3.
        result = restore_logits(values, config)
        self.assertEqual(result.shape, (2, 1, 101, 101))
        self.assertTrue(torch.all(result == 3))
        config.IMAGE_SIZE = 25
        config.PADDED_SIZE = 32
        values = torch.ones(1, 1, 32, 32)
        self.assertEqual(crop_padding(values, config).shape, (1, 1, 25, 25))
        self.assertEqual(restore_logits(values, config).shape, (1, 1, 101, 101))

    def test_flip_tta_restores_orientation(self):
        config = Config()
        config.DEVICE = 'cpu'
        predictor = ModelPredictor(config)
        images = torch.arange(32.).reshape(1, 1, 4, 8)
        self.assertTrue(torch.equal(predictor._predict_with_tta(torch.nn.Identity(), images), images))

    def test_losses_handle_empty_full_and_mixed_masks(self):
        targets = torch.stack([torch.zeros(1, 4, 4), torch.ones(1, 4, 4),
                               torch.eye(4).unsqueeze(0)])
        for name in ('lovasz', 'dice', 'bce', 'focal'):
            logits = torch.zeros_like(targets, requires_grad=True)
            loss = create_loss_function(name)(logits, targets)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertEqual(create_loss_function('lovasz')(targets * 4 - 2, targets).item(), 0)

    def test_paths_follow_configuration_without_side_effects(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as directory:
            root = Path(directory)
            config = Config(root / 'input', root / 'output')
            self.assertFalse((root / 'output').exists())
            config.DATA_DIR = root / 'other'
            self.assertEqual(config.RAW_DATA_DIR, root / 'other/download')
            config.OUTPUT_DIR = root / 'new_output'
            self.assertEqual(config.MODEL_DIR, root / 'new_output/models/seresnet34')
            with self.assertRaises(ValueError):
                ModelEvaluator(config).find_best_threshold()

    def test_checkpoint_can_save_zero_metric_and_resume_next_epoch(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as directory:
            config = Config(output_dir=directory)
            config.TINY_MODEL = True
            config.DEVICE = 'cpu'
            trainer = ModelTrainer(1, config)
            model, optimizer = trainer.create_model_and_optimizer('seresnet34')
            trainer.save_checkpoint(model, optimizer, epoch=0, metric=0.)
            self.assertEqual(trainer.load_checkpoint(model, optimizer), 1)
            self.assertEqual(trainer.best_metric, 0.)
            loaded = ModelPredictor(config).load_model(1, 'seresnet34')
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), loaded.parameters())))

    def test_each_production_backbone_forward_and_backward(self):
        config = Config()
        config.PRETRAINED = False
        for name in ('resnet34', 'seresnet34', 'vgg11'):
            with self.subTest(model=name):
                model = create_model(name, config).train()
                images = torch.randn(2, 3, 32, 32)
                output = model(images)
                self.assertEqual(output.shape, (2, 1, 32, 32))
                loss = create_loss_function('lovasz')(output, torch.zeros_like(output))
                loss.backward()
                self.assertTrue(torch.isfinite(model.head[-1].weight.grad).all())
                del model, output, loss


if __name__ == '__main__':
    unittest.main()
