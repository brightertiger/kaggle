"""Programmatic equivalent of main.py; accepts the same local data layout."""
import argparse
from pathlib import Path

from src.config import AvitoConfig, Config
from src.pipeline import AvitoPipeline


def run_example(data_dir=None, output_dir=None):
    defaults = AvitoConfig()
    output = Path(output_dir or defaults.OUTPUT_DIR)
    config = Config(AvitoConfig(
        DATA_ROOT=data_dir or defaults.DATA_ROOT,
        OUTPUT_DIR=str(output), FEATURES_DIR=str(output / 'features'),
        MODEL_DIR=str(output / 'models')))
    pipeline = AvitoPipeline(config)
    pipeline.preprocess_data()
    pipeline.generate_features()
    pipeline.train_models()
    print('Blend diagnostics:', pipeline.evaluate_pipeline())
    submission = pipeline.generate_submission()
    print('Feature inventory (not measured importance):', pipeline.get_feature_importance())
    return submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    run_example(args.data_dir, args.output_dir)
