#!/usr/bin/env python3
"""Run the synthetic example, or use the same API with competition CSVs."""
import argparse
from pathlib import Path

from src import Config, TalkingDataPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-data', action='store_true', help='Use competition CSVs')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--raw-data-dir', type=Path, default=None)
    args = parser.parse_args()
    if args.real_data:
        config = Config(args.data_dir, raw_data_dir=args.raw_data_dir)
        pipeline = TalkingDataPipeline(config)
        submission = pipeline.run_full_pipeline()
        print(pipeline.get_feature_importance().to_string(index=False))
    else:
        from dry_run import main as run_dry_run
        submission = run_dry_run()
    print(submission.head().to_string(index=False))


if __name__ == '__main__':
    main()
