"""Exercise the real RFCX pipeline on small synthetic audio, entirely on CPU."""
from pathlib import Path
import os

# Keep caches and generated files within this competition folder.
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / 'dry_run_output' / 'numba_cache'))
os.environ.setdefault('ALBUMENTATIONS_OFFLINE', '1')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.config import AudioConfig, Config, DataConfig, ModelConfig, TrainingConfig
from src.data_utils import AudioProcessor, TestDataset, ValidDataset
from src.pipeline import create_folds, resample_audio, run_full_pipeline, set_seed


def make_sample(data_dir: Path):
    rng = np.random.default_rng(2017)
    source_sr = 32000
    rows = []
    columns = ['recording_id', 'species_id', 'songtype_id', 't_min', 'f_min', 't_max', 'f_max']
    for split in ('train', 'test'):
        (data_dir / split).mkdir(parents=True, exist_ok=True)
    for index in range(8):
        recording_id = f'train_{index:03d}'
        species = index % 2
        frequency = 600 + species * 800
        t = np.arange(source_sr * 6) / source_sr
        audio = (0.1 * np.sin(2 * np.pi * frequency * t) + 0.005 * rng.normal(size=len(t))).astype(np.float32)
        sf.write(data_dir / 'train' / f'{recording_id}.flac', audio, source_sr)
        # Repeated annotations exercise recording-group separation across folds.
        for start, end in ((0.5, 1.5), (4.0, 5.0)):
            rows.append([recording_id, species, 0, start, frequency - 100, end, frequency + 100])
    pd.DataFrame(rows, columns=columns).to_csv(data_dir / 'train_tp.csv', index=False)
    pd.DataFrame(columns=columns).to_csv(data_dir / 'train_fp.csv', index=False)
    ids = ['test_short', 'test_long']
    for recording_id, duration in zip(ids, (2, 8)):
        t = np.arange(source_sr * duration) / source_sr
        audio = (0.1 * np.sin(2 * np.pi * 600 * t) + 0.005 * rng.normal(size=len(t))).astype(np.float32)
        sf.write(data_dir / 'test' / f'{recording_id}.flac', audio, source_sr)
    submission = pd.DataFrame({'recording_id': ids})
    for species in range(24):
        submission[f's{species}'] = 0.0
    submission.to_csv(data_dir / 'sample_submission.csv', index=False)
    return submission


def main():
    torch.set_num_threads(1)
    data_dir = ROOT / 'sample_data'
    output_dir = ROOT / 'dry_run_output'
    config = Config(
        audio=AudioConfig(sample_rate=8000, n_mels=64),
        model=ModelConfig(input_size=64, pretrained=False, backbone='mobilenetv3_small_050'),
        training=TrainingConfig(batch_size=4, epochs=1, num_folds=2, num_workers=0),
        data=DataConfig(train_data_path=str(output_dir / 'positive.csv'),
                        test_data_path=str(data_dir / 'sample_submission.csv'),
                        audio_data_path=str(output_dir / 'resample'),
                        model_save_path=str(output_dir / 'models'),
                        predictions_path=str(output_dir / 'predictions')),
        device='cpu',
    )
    set_seed(config.seed)
    template = make_sample(data_dir)
    create_folds(str(data_dir / 'train_tp.csv'), config.data.train_data_path,
                 config.training.num_folds, config.seed)
    folds = pd.read_csv(config.data.train_data_path)
    assert folds.groupby('recording_id')['fold'].nunique().eq(1).all()
    for split in ('train', 'test'):
        resample_audio(str(data_dir / split), str(output_dir / 'resample' / split), config.audio.sample_rate)

    # Verify short clips, deterministic validation, and window boundary coverage.
    processor = AudioProcessor(config)
    silence = processor.extract_audio_segment(np.zeros(100, dtype=np.float32), 0, 0.01)
    assert len(silence) == config.audio.sample_rate * config.audio.segment_length
    assert np.isfinite(processor.create_melspectrogram(silence)).all()
    valid = ValidDataset(folds, 1, config)
    assert torch.equal(valid[0][0], valid[0][0])
    test = TestDataset(template, config)
    assert test[0].shape == (1, 3, 64, 64)
    assert test[1].shape == (4, 3, 64, 64)

    run_full_pipeline(config, model_type='resnet', apply_tta=True, create_ensemble=True)
    for name in ('resnet_predictions', 'resnet_predictions_tta', 'ensemble_predictions'):
        path = output_dir / 'predictions' / f'{name}.csv'
        predictions = pd.read_csv(path)
        assert list(predictions.columns) == list(template.columns)
        assert predictions.recording_id.tolist() == template.recording_id.tolist()
        scores = predictions.iloc[:, 1:].to_numpy()
        assert scores.shape == (len(template), config.model.num_classes)
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    for fold in range(1, config.training.num_folds + 1):
        checkpoint = torch.load(output_dir / 'models' / 'resnet' / f'model_fold_{fold}.pt',
                                map_location='cpu', weights_only=True)
        assert checkpoint['epoch'] == 0 and np.isfinite(checkpoint['metric'])
    print('DRY RUN PASSED: synthetic FLAC → resampling → mel features → CPU training '
          '→ checkpoint reload → sliding-window/TTA prediction → rank ensemble.')
    print('8 train recordings, 2 test recordings, 24 output classes; 2 folds, 1 epoch each.')
    print(f'Submission: {output_dir / "predictions" / "ensemble_predictions.csv"}')
    print('No pipeline stages skipped. Random small backbone; no pretrained weights downloaded.')


if __name__ == '__main__':
    main()
