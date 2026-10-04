"""Command-line entry point for preprocessing, training, and submission generation."""
import argparse
from pathlib import Path

from src.config import AudioConfig, Config, DataConfig, ModelConfig, TrainingConfig
from src.pipeline import create_folds, resample_audio, run_full_pipeline, set_seed
from src.predictor import generate_predictions
from src.trainer import train_model


def main():
    parser = argparse.ArgumentParser(description='RFCX Species Audio Detection Pipeline')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'predict', 'full'], default='full')
    parser.add_argument('--model', choices=['resnet', 'resnest'], default='resnet')
    parser.add_argument('--data-dir', type=Path, default=Path('data'), help='Kaggle CSVs and train/test FLAC folders')
    parser.add_argument('--output-dir', type=Path, default=Path('output'), help='Prepared arrays, folds, models, predictions')
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default=None, help='Default: CUDA if available, otherwise CPU')
    parser.add_argument('--no-pretrained', action='store_true', help='Random initialization; no weight downloads')
    parser.add_argument('--backbone', default=None, help='Optional timm backbone override for smoke tests')
    parser.add_argument('--input-size', type=int, default=300)
    parser.add_argument('--sample-rate', type=int, default=32000)
    parser.add_argument('--n-mels', type=int, default=300)
    parser.add_argument('--segment-length', type=float, default=5)
    parser.add_argument('--overlap-ratio', type=float, default=0.75)
    args = parser.parse_args()

    config = Config(
        audio=AudioConfig(sample_rate=args.sample_rate, n_mels=args.n_mels,
                          segment_length=args.segment_length, overlap_ratio=args.overlap_ratio),
        model=ModelConfig(input_size=args.input_size, pretrained=not args.no_pretrained,
                          backbone=args.backbone),
        training=TrainingConfig(num_folds=args.folds, epochs=args.epochs,
                                batch_size=args.batch_size, learning_rate=args.lr,
                                num_workers=args.num_workers),
        data=DataConfig(train_data_path=str(args.output_dir / 'positive.csv'),
                        test_data_path=str(args.data_dir / 'sample_submission.csv'),
                        audio_data_path=str(args.output_dir / 'resample'),
                        model_save_path=str(args.output_dir / 'models'),
                        predictions_path=str(args.output_dir / 'predictions')),
        **({'device': args.device} if args.device else {}),
    )
    set_seed(config.seed)
    if args.mode == 'preprocess':
        create_folds(str(args.data_dir / 'train_tp.csv'), config.data.train_data_path,
                     config.training.num_folds, config.seed)
        for split in ('train', 'test'):
            resample_audio(str(args.data_dir / split), str(Path(config.data.audio_data_path) / split),
                           config.audio.sample_rate)
    elif args.mode == 'train':
        train_model(config, args.model)
    elif args.mode == 'predict':
        generate_predictions(config, args.model, args.tta)
    else:
        run_full_pipeline(config, args.model, args.tta, args.ensemble)


if __name__ == '__main__':
    main()
