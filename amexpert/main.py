#!/usr/bin/env python3

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import AmExpertPipeline


def main():
    parser = argparse.ArgumentParser(description='AmExpert Coupon Redemption Prediction Pipeline')
    parser.add_argument('--data-dir', default='data', help='Directory containing raw data')
    parser.add_argument('--feature-dir', default='data/feature', help='Directory for feature files')
    parser.add_argument('--model-dir', default='data/model', help='Directory for model files')
    parser.add_argument('--score-dir', default='data/score', help='Directory for score files')
    parser.add_argument('--step', choices=['preprocess', 'features', 'merge', 'train', 'blend', 'all'], 
                       default='all', help='Pipeline step to run')
    
    args = parser.parse_args()
    
    pipeline = AmExpertPipeline(
        data_dir=args.data_dir,
        feature_dir=args.feature_dir,
        model_dir=args.model_dir,
        score_dir=args.score_dir
    )
    
    if args.step == 'preprocess':
        pipeline.preprocess_data()
    elif args.step == 'features':
        pipeline.create_features()
    elif args.step == 'merge':
        pipeline.merge_features()
    elif args.step == 'train':
        pipeline.train_models()
    elif args.step == 'blend':
        pipeline.blend_predictions()
    elif args.step == 'all':
        pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()
