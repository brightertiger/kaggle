#!/usr/bin/env python3
"""Train fold models and write melanoma probabilities in competition format."""
import argparse
import numpy as np

from src.config import Config
from src.models import MelanomaClassifier, MelanomaClassifierV2
from src.pipeline import MelanomaPipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description='SIIM-ISIC Melanoma Classification Pipeline')
    parser.add_argument('--data_dir', default='data', help='CSV files plus train/ and test/ JPEG folders')
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--score_dir', default='scores')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS)
    parser.add_argument('--model_type', choices=['standard', 'v2'], default='standard')
    parser.add_argument('--model_name', default=Config.MODEL_NAME)
    parser.add_argument('--image_size', type=int, default=Config.IMAGE_SIZE)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--folds', type=int, default=Config.N_FOLDS)
    parser.add_argument('--num_workers', type=int, default=Config.NUM_WORKERS)
    parser.add_argument('--device', default=Config.DEVICE)
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True,
                        help='Disable to initialize EfficientNet without downloading weights')
    parser.add_argument('--use_tta', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--hair_mask_dir', default=None, help='Optional directory of grayscale hair masks')
    parser.add_argument('--ensemble_method', choices=['weighted_average'], default='weighted_average',
                        help='Equal-weight average of fold predictions')
    args = parser.parse_args(argv)
    config = Config(MODEL_NAME=args.model_name, IMAGE_SIZE=args.image_size,
                    BATCH_SIZE=args.batch_size, NUM_EPOCHS=args.epochs,
                    N_FOLDS=args.folds, NUM_WORKERS=args.num_workers,
                    DEVICE=args.device, PRETRAINED=args.pretrained,
                    HAIR_MASK_DIR=args.hair_mask_dir)
    model_class = MelanomaClassifier if args.model_type == 'standard' else MelanomaClassifierV2
    pipeline = MelanomaPipeline(args.data_dir, args.model_dir, args.score_dir, config=config)
    fold_scores, predictions = pipeline.run_full_pipeline(
        model_class=model_class, epochs=args.epochs, use_tta=args.use_tta,
        ensemble_method=args.ensemble_method)
    print(f'Pipeline completed: {len(predictions)} predictions')
    print(f'Cross-validation AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}')
    print(f'Models: {args.model_dir}; predictions: {args.score_dir}')


if __name__ == '__main__':
    main()
