"""Datasets for cached CT windows; augment only the training partition."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class IntracranialHemorrhageDataset(Dataset):
    def __init__(self, frame, config, split='train', augment=False):
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.cache = Path(config.OUTPUT_DIR) / 'images' / split
        ops = ([transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip()] if augment else
               [transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))])
        self.transform = transforms.Compose(ops + [transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with np.load(self.cache / f'{row["image"]}.npz') as cached:
            image = torch.from_numpy(cached['image'].copy())
        result = {'image': self.transform(image), 'idx': row['image']}
        if all(name in self.frame for name in self.config.CLASS_NAMES):
            result['label'] = torch.tensor(row[self.config.CLASS_NAMES].to_numpy(dtype=np.float32))
        return result


def create_data_loaders(fold_idx, config):
    frame = pd.read_csv(config.TRAIN_CSV, dtype={'patient': str})
    if set(frame['fold'].unique()) != set(range(1, config.NUM_FOLDS + 1)):
        raise ValueError('Cached folds do not match configuration; rerun preprocessing')
    train, valid = frame[frame.fold != fold_idx], frame[frame.fold == fold_idx]
    if train.empty or valid.empty or set(train.patient) & set(valid.patient):
        raise ValueError('Expected nonempty, patient-disjoint train/validation splits')
    generator = torch.Generator().manual_seed(config.SEED + fold_idx)
    return (DataLoader(IntracranialHemorrhageDataset(train, config, augment=True),
                       batch_size=config.BATCH_SIZE_TRAIN, shuffle=True,
                       num_workers=config.NUM_WORKERS, generator=generator),
            DataLoader(IntracranialHemorrhageDataset(valid, config),
                       batch_size=config.BATCH_SIZE_VALID, num_workers=config.NUM_WORKERS))


def create_inference_loader(config):
    return DataLoader(IntracranialHemorrhageDataset(pd.read_csv(config.TEST_CSV), config, split='test'),
                      batch_size=config.BATCH_SIZE_INFERENCE, num_workers=config.NUM_WORKERS)
