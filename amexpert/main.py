#!/usr/bin/env python3

import argparse
from src.pipeline import AmExpertPipeline


def main():
    parser = argparse.ArgumentParser(description='AmExpert Coupon Redemption Prediction Pipeline')
    parser.add_argument('--data-dir', default='data', help='Directory containing raw data')
    parser.add_argument('--feature-dir', default='data/feature', help='Directory for feature files')
    parser.add_argument('--model-dir', default='data/model', help='Directory for model files')
    parser.add_argument('--score-dir', default='data/score', help='Directory for score files')
    parser.add_argument('--step', choices=['preprocess', 'features', 'merge', 'train', 'blend', 'all'], 
                       default='all', help='Pipeline step to run')
    
    parser.add_argument('--validation-campaign-id', type=int, default=13)
    parser.add_argument('--num-boost-round', type=int, default=2000)
    parser.add_argument('--early-stopping-rounds', type=int, default=200)
    parser.add_argument('--num-threads', type=int, default=3)

    args = parser.parse_args()
    
    pipeline = AmExpertPipeline(
        data_dir=args.data_dir,
        feature_dir=args.feature_dir,
        model_dir=args.model_dir,
        score_dir=args.score_dir,
        validation_campaign_id=args.validation_campaign_id,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        model_params={"num_threads": args.num_threads}
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
