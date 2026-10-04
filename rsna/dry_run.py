#!/usr/bin/env python3
"""Exercise the actual CLI using synthetic DICOMs and no downloaded weights."""
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
import torch

from src.core import Config
from src.data import create_inference_loader, analyze_dataset
from src.data.preprocessing import window_image
from src.inference import ModelPredictor, load_trained_model
from src.models import create_model

ROOT = Path(__file__).resolve().parent


def write_dicom(path, patient, positive, rng):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientID = patient
    ds.Modality = 'CT'
    ds.Rows = ds.Columns = 512
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    pixels = rng.normal(1055, 12, (512, 512)).astype(np.int16)
    if positive:
        pixels[180:260, 230:310] += 60
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    rng = np.random.default_rng(Config.SEED)
    data = ROOT / 'sample_data'
    output = ROOT / 'dry_run_output'
    for split in ('train', 'test'):
        (data / split).mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(8):
        image_id = f'ID_{index:09d}'
        subtype = np.zeros(5, dtype=int)
        if index % 3:
            subtype[index % 5] = 1
        labels = np.r_[subtype.max(), subtype]
        write_dicom(data / 'train' / f'{image_id}.dcm', f'patient_{index // 2}', labels[0], rng)
        records.extend({'ID': f'{image_id}_{name}', 'Label': int(label)}
                       for name, label in zip(Config.CLASS_NAMES, labels))
    pd.DataFrame(records).to_csv(data / 'stage_1_train.csv', index=False)
    records = []
    for index in range(3):
        image_id = f'ID_{index + 100:09d}'
        write_dicom(data / 'test' / f'{image_id}.dcm', f'test_patient_{index}', index % 2, rng)
        records.extend({'ID': f'{image_id}_{name}', 'Label': 0.5} for name in Config.CLASS_NAMES)
    # Deliberately shuffled to verify that output follows the template, not sorted IDs.
    sample = pd.DataFrame(records).sample(frac=1, random_state=Config.SEED)
    sample.to_csv(data / 'stage_1_sample_submission.csv', index=False)
    common = [sys.executable, str(ROOT / 'main.py'), '--data-dir', str(data),
              '--output-dir', str(output), '--model', 'resnet18', '--device', 'cpu',
              '--epochs', '1', '--folds', '2', '--batch-size', '3', '--image-size', '64',
              '--workers', '0', '--no-pretrained', '--no-amp']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run(common + ['--mode', 'full', '--tta'], check=True, env=env)
    subprocess.run(common + ['--mode', 'validate'], check=True, env=env)

    config = Config(data, output)
    config.NUM_FOLDS, config.IMAGE_SIZE, config.NUM_WORKERS = 2, 64, 0
    config.DEVICE, config.PRETRAINED = 'cpu', False
    config.BATCH_SIZE_INFERENCE = 2
    manifest = pd.read_csv(config.TRAIN_CSV)
    assert manifest.groupby('patient').fold.nunique().eq(1).all()
    assert set(manifest.fold) == {1, 2}
    with np.load(output / 'images' / 'train' / 'ID_000000000.npz') as cached:
        assert cached['image'].shape == (3, 64, 64)
        assert 0 <= cached['image'].min() <= cached['image'].max() <= 1
    # Check slope/intercept conversion independently of the generated image cache.
    ds = pydicom.dcmread(data / 'train' / 'ID_000000000.dcm')
    ds.PixelData = np.full((512, 512), 1024, dtype=np.int16).tobytes()
    assert np.allclose(window_image(ds, config)[:, 0, 0], [0, 0.1, 150 / 380])
    submission = pd.read_csv(config.SCORE_DIR / 'submission.csv')
    assert submission.columns.tolist() == ['ID', 'Label']
    assert submission.ID.tolist() == sample.ID.tolist()
    assert submission.Label.between(0, 1).all() and np.isfinite(submission.Label).all()
    validation = pd.read_csv(output / 'combined_validation.csv')
    assert set(validation.image) == set(manifest.image) and len(validation) == len(manifest)
    for fold in (1, 2):
        checkpoint = torch.load(config.checkpoint_dir('resnet18', fold) / 'best_model.pt', weights_only=True)
        assert checkpoint['optimizer_state_dict']['state']
        assert np.isfinite(checkpoint['loss'])
        torch.manual_seed(config.SEED + fold)
        initial = create_model('resnet18', pretrained=False).state_dict()['backbone.fc.weight']
        assert not torch.equal(initial, checkpoint['model_state_dict']['backbone.fc.weight'])
    model = load_trained_model(config.checkpoint_dir('resnet18', 1) / 'best_model.pt', 'resnet18', config)
    predictor = ModelPredictor(model, 'cpu', config)
    loader = create_inference_loader(config)
    plain, augmented = predictor.predict_loader(loader), predictor.predict_with_tta(loader)
    assert plain['indices'] == augmented['indices']
    assert np.allclose(plain['predictions'], augmented['tta_predictions'][0], atol=1e-6)
    analyze_dataset(config)
    print('DRY RUN PASSED: 8 train / 3 test DICOM slices; 2 patient folds; 1 CPU epoch per fold.')
    print('Preprocessing, CNN features, training, checkpoint reload, validation, TTA and submission passed.')
    print(f'Submission: {config.SCORE_DIR / "submission.csv"}; skipped: none; pretrained downloads: none.')


if __name__ == '__main__':
    main()
