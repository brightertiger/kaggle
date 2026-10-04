"""Checkpoint inference, horizontal-flip TTA, fold averaging, and RLE."""
import json
import numpy as np
import pandas as pd
import torch

from ..models.models import create_model
from ..data.data_utils import create_data_loaders, create_test_loader, load_manifest, restore_logits


class ModelPredictor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)

    def load_model(self, fold_idx, model_name):
        self.config.MODEL_NAME = model_name
        model_path = self.config.MODEL_DIR / f'model_{fold_idx}.pth'
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        for key in ('MODEL_NAME', 'TINY_MODEL', 'IMAGE_SIZE', 'PADDED_SIZE', 'NUM_FOLDS', 'RANDOM_SEED'):
            if checkpoint['config'][key] != getattr(self.config, key):
                raise ValueError(f'Checkpoint configuration mismatch: {key}')
        # Loading a checkpoint never needs an ImageNet download.
        model = create_model(model_name, self.config, pretrained=False).to(self.device)
        model.load_state_dict(checkpoint['state_dict'])
        return model.eval()

    def predict_single_fold(self, fold_idx, model_name, use_tta=False):
        model = self.load_model(fold_idx, model_name)
        self.config._create_directories()
        _, valid_loader = create_data_loaders(fold_idx, self.config)
        test_loader = create_test_loader(self.config)
        valid_scores, valid_actuals = self._predict_loader(model, valid_loader, use_tta)
        test_scores = self._predict_loader(model, test_loader, use_tta, has_masks=False)
        np.save(self.config.SCORES_DIR / 'valid' / f'scores_{fold_idx}.npy', valid_scores)
        np.save(self.config.SCORES_DIR / 'valid' / f'actuals_{fold_idx}.npy', valid_actuals)
        np.save(self.config.SCORES_DIR / 'test' / f'scores_{fold_idx}.npy', test_scores)
        metadata = {'use_tta': use_tta, 'test_ids': test_loader.dataset.frame.id.tolist(),
                    'config': self.config.to_dict(),
                    'checkpoint_mtime': (self.config.MODEL_DIR / f'model_{fold_idx}.pth').stat().st_mtime_ns}
        (self.config.SCORES_DIR / f'fold_{fold_idx}.json').write_text(json.dumps(metadata))
        return valid_scores, test_scores

    def _predict_loader(self, model, data_loader, use_tta=False, has_masks=True):
        predictions, actuals = [], []
        with torch.no_grad():
            for sample in data_loader:
                images = sample['image'].to(self.device)
                preds = self._predict_with_tta(model, images) if use_tta else model(images)
                predictions.append(restore_logits(preds, self.config).cpu().numpy())
                if has_masks:
                    actuals.append(sample['original_mask'].numpy())
        if not predictions:
            raise ValueError('Cannot predict an empty dataset')
        scores = np.concatenate(predictions)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite model predictions')
        return (scores, np.concatenate(actuals)) if has_masks else scores

    def _predict_with_tta(self, model, images):
        original = model(images)
        flipped = torch.flip(model(torch.flip(images, dims=[3])), dims=[3])
        return (original + flipped) / 2

    def predict_all_folds(self, model_name, use_tta=False):
        return {fold: self.predict_single_fold(fold, model_name, use_tta)[1]
                for fold in range(1, self.config.NUM_FOLDS + 1)}

    def create_submission(self, model_name, use_tta=False, threshold=None):
        self.config.MODEL_NAME = model_name
        self.config._create_directories()
        threshold = self.config.IOU_CUTOFF if threshold is None else threshold
        test_ids = load_manifest(self.config, 'test').id.tolist()
        all_predictions = []
        for fold in range(1, self.config.NUM_FOLDS + 1):
            scores_path = self.config.SCORES_DIR / 'test' / f'scores_{fold}.npy'
            metadata_path = self.config.SCORES_DIR / f'fold_{fold}.json'
            if not scores_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(f'Missing fold {fold} predictions; run predict first')
            metadata = json.loads(metadata_path.read_text())
            model_path = self.config.MODEL_DIR / f'model_{fold}.pth'
            if metadata['use_tta'] != use_tta or metadata['test_ids'] != test_ids:
                raise ValueError('Prediction IDs or TTA settings changed; rerun predict')
            for key in ('IMAGE_SIZE', 'PADDED_SIZE', 'ORIGINAL_SIZE', 'NUM_FOLDS', 'RANDOM_SEED', 'MEAN', 'STD'):
                if metadata['config'][key] != getattr(self.config, key):
                    raise ValueError(f'Prediction configuration changed: {key}; rerun predict')
            if not model_path.exists() or metadata['checkpoint_mtime'] != model_path.stat().st_mtime_ns:
                raise ValueError('Model changed since prediction; rerun predict')
            scores = np.load(scores_path)
            expected = (len(test_ids), 1, self.config.ORIGINAL_SIZE, self.config.ORIGINAL_SIZE)
            if scores.shape != expected or not np.isfinite(scores).all():
                raise ValueError(f'Invalid prediction array for fold {fold}')
            all_predictions.append(scores)
        ensemble = np.mean(all_predictions, axis=0)
        rles = []
        for scores in ensemble[:, 0]:
            mask = scores >= threshold
            if mask.sum() <= self.config.MIN_SALT_PIXELS:
                mask[:] = False
            rles.append(self._rle_encode(mask))
        submission = pd.DataFrame({'id': test_ids, 'rle_mask': rles})
        path = self.config.SUBMIT_DIR / f'{self.config.MODEL_TAG}_submission.csv'
        submission.to_csv(path, index=False)
        print(f'Submission saved to {path}')
        return submission

    @staticmethod
    def _rle_encode(image):
        pixels = np.concatenate(([0], np.asarray(image).flatten(order='F'), [0]))
        runs = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
        runs[1::2] -= runs[::2]
        return ' '.join(map(str, runs))
