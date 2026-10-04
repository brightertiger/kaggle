"""Convert competition DICOMs to windowed arrays and patient-disjoint folds."""
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.model_selection import GroupKFold
from tqdm import tqdm


def parse_ids(frame, classes):
    if not {'ID', 'Label'}.issubset(frame.columns) or frame.empty:
        raise ValueError('Expected a nonempty CSV with ID and Label columns')
    parts = frame['ID'].str.rsplit('_', n=1, expand=True)
    if parts.shape[1] != 2 or not parts[1].isin(classes).all():
        raise ValueError('Unknown hemorrhage class in ID')
    if not parts[0].str.fullmatch(r'ID_[A-Za-z0-9]+').all():
        raise ValueError('Expected IDs such as ID_000000001_any')
    result = frame.assign(image=parts[0], category=parts[1])
    if result.duplicated(['image', 'category']).any():
        raise ValueError('Duplicate image/class labels')
    return result


def window_image(dicom, config):
    pixels = dicom.pixel_array.astype(np.float32)
    if pixels.ndim != 2:
        raise ValueError('Expected a single grayscale CT slice')
    hu = pixels * float(dicom.get('RescaleSlope', 1)) + float(dicom.get('RescaleIntercept', 0))
    windows = [np.clip((hu - (center - width / 2)) / width, 0, 1)
               for center, width in zip(config.WINDOW_CENTERS, config.WINDOW_WIDTHS)]
    image = torch.from_numpy(np.stack(windows))
    image = torch.nn.functional.interpolate(image[None], size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                                           mode='bilinear', align_corners=False)[0]
    return image.numpy().astype(np.float32)


def preprocess_all_data(config):
    config.ensure_output_dirs()
    labels = parse_ids(pd.read_csv(Path(config.DATA_DIR) / config.TRAIN_LABELS), config.CLASS_NAMES)
    train = labels.pivot(index='image', columns='category', values='Label').reindex(columns=config.CLASS_NAMES)
    if train.isna().any().any() or not train.isin([0, 1]).all().all():
        raise ValueError('Each training image needs all six binary labels')
    train = train.reset_index()
    if not (train['any'] == train[config.CLASS_NAMES[1:]].max(axis=1)).all():
        raise ValueError('The any label must agree with the subtype labels')
    sample = parse_ids(pd.read_csv(Path(config.DATA_DIR) / config.SAMPLE_SUBMISSION), config.CLASS_NAMES)
    if not sample.groupby('image').size().eq(config.NUM_CLASSES).all():
        raise ValueError('Each test image needs all six submission IDs')
    test = sample[['image']].drop_duplicates().reset_index(drop=True)
    for split, frame, directory in [('train', train, config.TRAIN_DIR), ('test', test, config.TEST_DIR)]:
        cache = Path(config.OUTPUT_DIR) / 'images' / split
        cache.mkdir(parents=True, exist_ok=True)
        patients = []
        for image_id in tqdm(frame['image'], desc=f'Preprocess {split}'):
            dicom = pydicom.dcmread(directory / f'{image_id}.dcm')
            if split == 'train':
                patient = str(dicom.get('PatientID', '')).strip()
                if not patient:
                    raise ValueError(f'Missing PatientID in {image_id}; cannot split safely')
                patients.append(patient)
            np.savez_compressed(cache / f'{image_id}.npz', image=window_image(dicom, config))
        if split == 'train':
            frame['patient'] = patients
    if not 2 <= config.NUM_FOLDS <= train['patient'].nunique():
        raise ValueError('NUM_FOLDS must be between 2 and the number of training patients')
    train['fold'] = 0
    splitter = GroupKFold(n_splits=config.NUM_FOLDS)
    for fold, (_, valid) in enumerate(splitter.split(train, groups=train['patient']), 1):
        train.loc[valid, 'fold'] = fold
    train.to_csv(config.TRAIN_CSV, index=False)
    test.to_csv(config.TEST_CSV, index=False)
