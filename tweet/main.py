#!/usr/bin/env python3
"""Train, evaluate processed folds, or write a competition submission."""
import argparse
from pathlib import Path
from src.config import get_config
from src.pipeline import TweetSentimentPipeline


def main():
    parser = argparse.ArgumentParser(description='Tweet Sentiment Extraction Pipeline')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], default='train')
    parser.add_argument('--data-path', help='Raw train CSV for train; processed fold CSV for evaluate')
    parser.add_argument('--test-path', help='Test CSV with textID, text, sentiment')
    parser.add_argument('--output-path', default='submissions/', help='Submission output directory')
    parser.add_argument('--config', help='JSON file with data/model/training overrides')
    parser.add_argument('--device', help='Override training.device, e.g. cpu or cuda:0')
    args = parser.parse_args()
    config = get_config(args.config)
    if args.device:
        config.training.device = args.device
    if args.data_path:
        config.data.train_path = args.data_path
    if args.test_path:
        config.data.test_path = args.test_path
    pipeline = TweetSentimentPipeline(config)
    if args.mode == 'train':
        pipeline.run_full_pipeline(config.data.train_path)
    elif args.mode == 'evaluate':
        results = pipeline.evaluate_all_folds(config.data.train_path)
        print(f"Average Jaccard Score: {sum(results.values()) / len(results):.4f}")
    else:
        submission = pipeline.predict_test_set(config.data.test_path)
        output = Path(args.output_path)
        output.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output / 'submission.csv', index=False)
        print(f"Predictions saved to: {output / 'submission.csv'}")


if __name__ == '__main__':
    main()
