"""Prepare the official train.csv without copying or modifying source images."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def validate_image_ids(data):
    if 'image_id' not in data or data.empty or data['image_id'].isna().any():
        raise ValueError('Expected nonempty image_id column')
    if data['image_id'].duplicated().any():
        raise ValueError('image_id values must be unique')
    if any(not isinstance(value, str) or Path(value).name != value for value in data['image_id']):
        raise ValueError('image_id values must be filenames, not paths')


def prepare_data(data_dir, output_path=None, n_splits=5, seed=42, num_classes=5):
    data_dir = Path(data_dir)
    data = pd.read_csv(data_dir / 'train.csv')
    validate_image_ids(data)
    if 'label' not in data:
        raise ValueError('train.csv must contain image_id and label')
    labels = pd.to_numeric(data['label'], errors='raise')
    if labels.isna().any() or not np.all(labels == labels.astype(int)):
        raise ValueError('Labels must be integers')
    data['label'] = labels.astype(int)
    if not data['label'].between(0, num_classes - 1).all():
        raise ValueError('Labels are outside the configured class range')
    counts = data['label'].value_counts()
    if len(counts) != num_classes or counts.min() < n_splits:
        raise ValueError('Each class must have at least n_splits training images')
    for image_id in data['image_id']:
        if not (data_dir / 'train_images' / image_id).is_file():
            raise FileNotFoundError(data_dir / 'train_images' / image_id)
    data['fold'] = -1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(data, data['label'])):
        data.loc[valid_idx, 'fold'] = fold
    output_path = Path(output_path) if output_path else data_dir / 'folds.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


def analyze_data(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(data)
    counts = data.groupby(['fold', 'label']).size().unstack(fill_value=0)
    print('Class counts by fold:\n', counts)
    return counts
