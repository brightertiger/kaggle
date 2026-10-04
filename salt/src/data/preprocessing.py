"""Validate raw PNGs and assign coverage-stratified validation folds."""
import json
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold


def coverage_class(mask):
    """Sparse, vertically uniform, then increasing foreground coverage."""
    if mask.sum() < 8:
        return 0
    if np.all(mask == mask[:1, :]):
        return 1
    return int(np.digitize(mask.mean(), [0.15, 0.25, 0.50, 0.67])) + 2


class SaltDataProcessor:
    def __init__(self, config):
        self.config = config

    def run_preprocessing(self):
        cfg = self.config
        cfg._create_directories()
        raw = cfg.RAW_DATA_DIR
        train = pd.read_csv(raw / 'train.csv', dtype={'id': str}, keep_default_na=False)
        if not {'id', 'rle_mask'}.issubset(train.columns):
            raise ValueError('train.csv must contain id and rle_mask')
        if train.id.duplicated().any() or len(train) < cfg.NUM_FOLDS:
            raise ValueError('Training IDs must be unique with at least NUM_FOLDS images')
        depths = pd.read_csv(raw / 'depths.csv', dtype={'id': str})
        if not {'id', 'z'}.issubset(depths.columns) or depths.id.duplicated().any():
            raise ValueError('depths.csv must contain unique id and z columns')
        sample_path = raw / 'sample_submission.csv'
        if sample_path.exists():
            test = pd.read_csv(sample_path, dtype={'id': str}, keep_default_na=False)
            if not {'id', 'rle_mask'}.issubset(test.columns):
                raise ValueError('sample_submission.csv must contain id and rle_mask')
            test = test[['id']]
        else:
            test = pd.DataFrame({'id': sorted(p.stem for p in (raw / 'test/images').glob('*.png'))})
        if test.empty or test.id.duplicated().any() or set(train.id) & set(test.id):
            raise ValueError('Test IDs must be nonempty, unique, and separate from training')
        classes, coverages = [], []
        for split, frame in [('train', train), ('test', test)]:
            for image_id in frame.id:
                if not image_id or '/' in image_id or '\\' in image_id or image_id in {'.', '..'}:
                    raise ValueError(f'Invalid image ID: {image_id!r}')
                with Image.open(raw / split / 'images' / f'{image_id}.png') as image:
                    if image.size != (cfg.ORIGINAL_SIZE, cfg.ORIGINAL_SIZE):
                        raise ValueError(f'Unexpected image shape for {image_id}: {image.size}')
                if split == 'train':
                    with Image.open(raw / 'train/masks' / f'{image_id}.png') as image:
                        mask = np.asarray(image.convert('L')) > 0
                    if mask.shape != (cfg.ORIGINAL_SIZE, cfg.ORIGINAL_SIZE):
                        raise ValueError(f'Unexpected mask shape for {image_id}')
                    classes.append(coverage_class(mask))
                    coverages.append(float(mask.mean()))
        all_ids = set(train.id) | set(test.id)
        if not all_ids.issubset(set(depths.id)):
            raise ValueError('depths.csv is missing image IDs')
        train['coverage'] = coverages
        train['coverage_class'] = classes
        train = train.merge(depths[['id', 'z']], on='id', validate='one_to_one')
        test = test.merge(depths[['id', 'z']], on='id', validate='one_to_one')
        labels = train.coverage_class
        # Merge rare strata for small datasets; explicitly fall back if still impossible.
        counts = labels.value_counts()
        labels = labels.where(labels.map(counts) >= cfg.NUM_FOLDS, -1)
        if labels.value_counts().min() < cfg.NUM_FOLDS:
            warnings.warn('Insufficient examples per coverage stratum; using seeded KFold.', stacklevel=2)
            splitter = KFold(cfg.NUM_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED)
            split_kind = 'kfold'
        else:
            splitter = StratifiedKFold(cfg.NUM_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED)
            split_kind = 'coverage-stratified'
        train['fold'] = 0
        for fold, (_, valid) in enumerate(splitter.split(train, labels), start=1):
            train.loc[valid, 'fold'] = fold
        train.to_csv(cfg.PROCESSED_DATA_DIR / 'train.csv', index=False)
        test.to_csv(cfg.PROCESSED_DATA_DIR / 'test.csv', index=False)
        (cfg.PROCESSED_DATA_DIR / 'metadata.json').write_text(json.dumps({
            'raw_dir': str(raw.resolve()), 'num_folds': cfg.NUM_FOLDS,
            'original_size': cfg.ORIGINAL_SIZE, 'random_seed': cfg.RANDOM_SEED,
            'split': split_kind,
        }, indent=2))
        print(f'Prepared {len(train)} train / {len(test)} test images ({split_kind}).')
