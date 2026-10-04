"""Command-line entry point; all generated files default to ./output."""
import argparse
from src.pipeline import CassavaPipeline
from src.utils.config import Config


def main():
    parser = argparse.ArgumentParser(description='Cassava Leaf Disease Classification')
    parser.add_argument('--mode', choices=['prepare_data', 'train', 'score', 'ensemble', 'full_pipeline'], default='train')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--version', default='version7', help='Run label for train/score')
    parser.add_argument('--versions', nargs='+', default=None, help='Run labels for full_pipeline/ensemble; defaults to --version')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--test_path', default=None, help='CSV with image_id; defaults to DATA_DIR/sample_submission.csv')
    parser.add_argument('--model_name', default=Config.MODEL_NAME)
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--device', default=Config.DEVICE)
    parser.add_argument('--image_size', type=int, default=Config.IMAGE_SIZE)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--num_workers', type=int, default=Config.NUM_WORKERS)
    parser.add_argument('--n_folds', type=int, default=Config.N_FOLDS)
    parser.add_argument('--seed', type=int, default=Config.SEED)
    parser.add_argument('--learning_rate', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--accumulation_steps', type=int, default=Config.ACCUMULATION_STEPS)
    parser.add_argument('--swa_start', type=int, default=Config.SWA_START)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--num_tta', type=int, default=Config.NUM_TTA)
    args = parser.parse_args()
    config = Config(DATA_DIR=args.data_dir, OUTPUT_DIR=args.output_dir,
                    MODEL_NAME=args.model_name, PRETRAINED=args.pretrained, DEVICE=args.device,
                    IMAGE_SIZE=args.image_size, BATCH_SIZE=args.batch_size, EPOCHS=args.epochs,
                    NUM_WORKERS=args.num_workers, N_FOLDS=args.n_folds, SEED=args.seed,
                    LEARNING_RATE=args.learning_rate, ACCUMULATION_STEPS=args.accumulation_steps,
                    SWA_START=args.swa_start, PATIENCE=args.patience, NUM_TTA=args.num_tta,
                    VERSIONS=tuple(args.versions or [args.version]))
    pipeline = CassavaPipeline(config)
    if args.mode == 'prepare_data':
        pipeline.prepare_data()
    elif args.mode == 'train':
        pipeline.train_model(args.version, args.fold)
    elif args.mode == 'score':
        pipeline.score_model(args.version, args.test_path)
    elif args.mode == 'ensemble':
        pipeline.create_ensemble(test_path=args.test_path)
    else:
        pipeline.run_full_pipeline(test_path=args.test_path)


if __name__ == '__main__':
    main()
