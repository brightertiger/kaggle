"""PNG datasets, paired augmentation, and reversible model geometry."""
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def crop_padding(values, config):
    start = config.PAD_BEFORE
    return values[..., start:start + config.IMAGE_SIZE, start:start + config.IMAGE_SIZE]


def restore_logits(values, config):
    values = crop_padding(values, config)
    return F.interpolate(values, size=(config.ORIGINAL_SIZE, config.ORIGINAL_SIZE),
                         mode='bilinear', align_corners=False)


def load_manifest(config, split):
    metadata = json.loads((config.PROCESSED_DATA_DIR / 'metadata.json').read_text())
    expected = {'raw_dir': str(config.RAW_DATA_DIR.resolve()), 'num_folds': config.NUM_FOLDS,
                'original_size': config.ORIGINAL_SIZE, 'random_seed': config.RANDOM_SEED}
    if any(metadata[key] != value for key, value in expected.items()):
        raise ValueError('Preprocessing configuration changed; rerun preprocessing')
    return pd.read_csv(config.PROCESSED_DATA_DIR / f'{split}.csv', dtype={'id': str},
                       keep_default_na=False)


class SaltDataset(Dataset):
    def __init__(self, frame, config, split='train', augment=False):
        self.frame = frame.reset_index(drop=True)
        self.config, self.split, self.augment = config, split, augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        cfg = self.config
        image_id = self.frame.iloc[index]['id']
        with Image.open(cfg.RAW_DATA_DIR / self.split / 'images' / f'{image_id}.png') as source:
            image = np.array(source.convert('RGB').resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                                                         Image.Resampling.BILINEAR))
        mask = original = None
        if self.split == 'train':
            with Image.open(cfg.RAW_DATA_DIR / 'train/masks' / f'{image_id}.png') as source:
                original = (np.array(source.convert('L')) > 0).astype(np.float32)
                mask = (np.array(source.convert('L').resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                                                           Image.Resampling.NEAREST)) > 0).astype(np.float32)
        if self.augment and cfg.USE_FLIP_AUGMENTATION and torch.rand(()).item() < cfg.FLIP_PROBABILITY:
            image = image[:, ::-1]
            mask = mask[:, ::-1]
            original = original[:, ::-1]
        before = cfg.PAD_BEFORE
        after = cfg.PADDED_SIZE - cfg.IMAGE_SIZE - before
        image = np.pad(image, ((before, after), (before, after), (0, 0)), mode='reflect')
        image = (image.astype(np.float32) / 255 - np.array(cfg.MEAN, dtype=np.float32)) / np.array(cfg.STD, dtype=np.float32)
        sample = {'id': image_id, 'image': torch.from_numpy(image.transpose(2, 0, 1).copy())}
        if mask is not None:
            mask = np.pad(mask, ((before, after), (before, after)), mode='constant')
            sample['mask'] = torch.from_numpy(mask[None].copy())
            sample['original_mask'] = torch.from_numpy(original[None].copy())
        return sample


def create_data_loaders(fold_idx, config):
    config.validate()
    if not 1 <= fold_idx <= config.NUM_FOLDS:
        raise ValueError('Fold is outside configured range')
    frame = load_manifest(config, 'train')
    train, valid = frame[frame.fold != fold_idx], frame[frame.fold == fold_idx]
    if train.empty or valid.empty:
        raise ValueError('Both training and validation splits must contain images')
    generator = torch.Generator().manual_seed(config.RANDOM_SEED + fold_idx)
    return (
        DataLoader(SaltDataset(train, config, augment=True), batch_size=config.BATCH_SIZE_TRAIN,
                   shuffle=True, generator=generator, num_workers=config.NUM_WORKERS),
        DataLoader(SaltDataset(valid, config), batch_size=config.BATCH_SIZE_VALID,
                   shuffle=False, num_workers=config.NUM_WORKERS),
    )


def create_test_loader(config):
    return DataLoader(SaltDataset(load_manifest(config, 'test'), config, split='test'),
                      batch_size=config.BATCH_SIZE_VALID, shuffle=False, num_workers=config.NUM_WORKERS)
