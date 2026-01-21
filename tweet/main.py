#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.config import get_config
from src.pipeline import TweetSentimentPipeline

def main():
    parser = argparse.ArgumentParser(description='Tweet Sentiment Analysis Pipeline')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict'], 
                       default='train', help='Pipeline mode')
    parser.add_argument('--data-path', type=str, default='data/raw/train.csv',
                       help='Path to training data')
    parser.add_argument('--test-path', type=str, default='data/raw/test.csv',
                       help='Path to test data')
    parser.add_argument('--output-path', type=str, default='submissions/',
                       help='Path to save predictions')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file')
    
    args = parser.parse_args()
    
    config = get_config()
    pipeline = TweetSentimentPipeline(config)
    
    if args.mode == 'train':
        print("Training mode selected")
        results = pipeline.run_full_pipeline(args.data_path)
        
        print("\nTraining completed!")
        print(f"Average Jaccard Score: {results['average_score']:.4f}")
        
    elif args.mode == 'evaluate':
        print("Evaluation mode selected")
        results = pipeline.evaluate_all_folds(args.data_path)
        
        avg_score = sum(results.values()) / len(results)
        print(f"\nAverage Jaccard Score: {avg_score:.4f}")
        
    elif args.mode == 'predict':
        print("Prediction mode selected")
        
        if not os.path.exists(args.test_path):
            print(f"Test file not found: {args.test_path}")
            return
        
        os.makedirs(args.output_path, exist_ok=True)
        
        submission = pipeline.predict_test_set(args.test_path)
        output_file = os.path.join(args.output_path, 'submission.csv')
        submission.to_csv(output_file, index=False)
        
        print(f"Predictions saved to: {output_file}")

if __name__ == '__main__':
    main()
