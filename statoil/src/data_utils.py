"""Radar channel transforms and stratified, fold-local angle preprocessing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold


class DataProcessor:
    def __init__(self, config):
        self.config = config

    def normalize_array(self, array):
        std = array.std()
        return (array - array.mean()) / (std if std > 0 else 1.0)

    def _convert_images(self, dataframe, source):
        images = []
        for _, row in dataframe.iterrows():
            horizontal = np.asarray(row['band_1'], dtype=np.float32).reshape(75, 75)
            vertical = np.asarray(row['band_2'], dtype=np.float32).reshape(75, 75)
            if source == 1:
                channels = (np.abs(vertical - horizontal), np.maximum(vertical, horizontal),
                            np.minimum(vertical, horizontal))
            else:
                channels = (vertical, horizontal, (vertical + horizontal) / 2)
            image = np.dstack([self.normalize_array(channel) for channel in channels])
            if self.config.IMAGE_SIZE != 75:
                scale = self.config.IMAGE_SIZE / 75
                image = zoom(image, (scale, scale, 1), order=1)
            images.append(image)
        return np.asarray(images, dtype=np.float32)

    def convert_images_source1(self, dataframe):
        return self._convert_images(dataframe, 1)

    def convert_images_source2(self, dataframe):
        return self._convert_images(dataframe, 2)

    def process_angles(self, angles, stats=None):
        values = pd.to_numeric(pd.Series(angles), errors='coerce').to_numpy(dtype=float, copy=True)
        values[~np.isfinite(values)] = np.nan
        if stats is None:
            observed = values[np.isfinite(values)]
            fill = float(observed.mean()) if len(observed) else 0.0
            filled = np.nan_to_num(values, nan=fill)
            stats = {'fill': fill, 'min': float(filled.min()),
                     'range': float(filled.max() - filled.min()) or 1.0}
        filled = np.nan_to_num(values, nan=stats['fill'])
        return ((filled - stats['min']) / stats['range']).astype(np.float32), stats

    def create_folds(self, images, labels, angles, ids, source_name):
        folds = StratifiedKFold(n_splits=self.config.FOLDS,
                               random_state=self.config.RANDOM_STATE, shuffle=True)
        folder = Path(self.config.DATA_DIR) / source_name / 'train'
        folder.mkdir(parents=True, exist_ok=True)
        angles = np.asarray(angles)
        for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(images, labels), 1):
            train_angles, stats = self.process_angles(angles[train_idx])
            valid_angles, _ = self.process_angles(angles[valid_idx], stats)
            (folder / f'angle_stats_{fold_idx}.json').write_text(json.dumps(stats))
            for prefix, indices, transformed in (
                ('train', train_idx, train_angles), ('test', valid_idx, valid_angles)
            ):
                for name, values in (('images', images[indices]), ('labels', labels[indices]),
                                     ('angles', transformed), ('ids', ids[indices])):
                    np.save(folder / f'{prefix}_{name}_{fold_idx}.npy', values)

    def process_train_data(self, source_name, convert_func):
        data = pd.read_json(Path(self.config.DATA_DIR) / 'download/train.json')
        images = convert_func(data)
        angles = data['inc_angle'].to_numpy()
        labels = data['is_iceberg'].to_numpy(dtype=np.float32)
        ids = data['id'].to_numpy(dtype=str)
        self.create_folds(images, labels, angles, ids, source_name)
        return images, angles, labels, ids

    def process_test_data(self, source_name, convert_func):
        data = pd.read_json(Path(self.config.DATA_DIR) / 'download/test.json')
        images = convert_func(data)
        ids = data['id'].to_numpy(dtype=str)
        folder = Path(self.config.DATA_DIR) / source_name / 'score'
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'images.npy', images)
        np.save(folder / 'ids.npy', ids)
        all_angles = []
        for fold_idx in range(1, self.config.FOLDS + 1):
            stats_file = folder.parent / 'train' / f'angle_stats_{fold_idx}.json'
            stats = json.loads(stats_file.read_text())
            angles, _ = self.process_angles(data['inc_angle'], stats)
            np.save(folder / f'angles_{fold_idx}.npy', angles)
            all_angles.append(angles)
        return images, np.asarray(all_angles), ids
