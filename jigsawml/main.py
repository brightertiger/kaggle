import argparse
from pathlib import Path
from src.pipeline import JigsawPipeline
from src.utils.config import Config


def main():
    parser = argparse.ArgumentParser(description='Jigsaw Multilingual Toxic Comment Classification')
    parser.add_argument('--mode', choices=['prepare_data', 'embeddings', 'adversarial', 'train',
                                          'score', 'ensemble', 'full_pipeline'], default='train')
    parser.add_argument('--subset', type=int, default=0, help='English training partition')
    parser.add_argument('--version', type=int, choices=[1, 2], default=1)
    parser.add_argument('--load_pretrained', action='store_true', help='Initialize v1 from its existing checkpoint')
    parser.add_argument('--from_scratch', action='store_true', help='Do not initialize v2 from v1')
    parser.add_argument('--data_dir', default='data', help='Contains raw/ and generated process/')
    parser.add_argument('--model_dir', default='model')
    parser.add_argument('--test_path', help='Defaults to processed foreign test data')
    parser.add_argument('--device', default=Config.DEVICE)
    parser.add_argument('--folds', type=int, default=Config.N_FOLDS)
    parser.add_argument('--epochs_v1', type=int, default=Config.EPOCHS_V1)
    parser.add_argument('--epochs_v2', type=int, default=Config.EPOCHS_V2)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--model_name', default=Config.MODEL_NAME)
    parser.add_argument('--tiny', action='store_true', help='Offline random XLM-R and smoke-test tokenizer')
    args = parser.parse_args()
    config = Config(args.data_dir, args.model_dir, DEVICE=args.device, N_FOLDS=args.folds,
                    EPOCHS_V1=args.epochs_v1, EPOCHS_V2=args.epochs_v2, BATCH_SIZE=args.batch_size,
                    MODEL_NAME=args.model_name, TINY=args.tiny)
    pipeline = JigsawPipeline(config)
    test_path = args.test_path or str(Path(config.DATA_DIR) / 'foreign/test_foreign.csv')
    if args.mode == 'prepare_data':
        pipeline.prepare_data(args.data_dir)
    elif args.mode == 'embeddings':
        pipeline.generate_embeddings(args.data_dir)
    elif args.mode == 'adversarial':
        pipeline.generate_adversarial_data(args.data_dir)
    elif args.mode == 'train':
        if args.version == 1:
            pipeline.train_version1(args.subset, args.load_pretrained)
        else:
            pipeline.train_version2(args.subset, load_from_version1=not args.from_scratch)
    elif args.mode == 'score':
        pipeline.scoring_pipeline.score_all_models(test_path)
    elif args.mode == 'ensemble':
        pipeline.create_ensemble()
    else:
        pipeline.run_full_pipeline(args.data_dir, test_path)


if __name__ == '__main__':
    main()
