#!/usr/bin/env python3

import argparse
from pathlib import Path
from src.pipeline import QuestionUnderstandingPipeline
from src.config import Config


def main():
    parser = argparse.ArgumentParser(description="Question Understanding Model Training and Inference")
    parser.add_argument("--mode", choices=["train", "inference", "evaluate"], required=True, help="Train, evaluate held-out folds, or predict test rows")
    parser.add_argument("--data_path", type=str, default=None, help="Path to data directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to save/load models")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--fold", type=int, default=None, help="Fold number for training (1 through n_folds)")

    parser.add_argument("--output_path", default=None, help="Submission directory")
    parser.add_argument("--device", default=None, help="auto, cpu, or cuda:0")
    args = parser.parse_args()

    config = Config()
    config_path = args.config
    if config_path is None and args.mode != 'train':
        saved = Path(args.model_path or config.model_dir) / 'config.json'
        if saved.exists():
            config_path = str(saved)
    if config_path:
        config.load_config(config_path)

    if args.device is not None:
        config.device = args.device
    data_path = args.data_path or config.data_dir
    model_path = args.model_path or config.model_dir
    config.data_dir = data_path
    config.model_dir = model_path
    if args.output_path is not None:
        config.output_dir = args.output_path
    pipeline = QuestionUnderstandingPipeline(config)

    if args.mode == "train":
        config.save_config(f"{model_path}/config.json")
        if args.fold is not None:
            pipeline.train_fold(args.fold, data_path, model_path)
        else:
            pipeline.train_all_folds(data_path, model_path)
    elif args.mode == "inference":
        pipeline.inference(data_path, model_path, args.output_path)
    elif args.mode == "evaluate":
        pipeline.evaluate_cv_performance(data_path, model_path)


if __name__ == "__main__":
    main()
