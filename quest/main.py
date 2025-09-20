#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline import QuestionUnderstandingPipeline
from src.config import Config


def main():
    parser = argparse.ArgumentParser(description="Question Understanding Model Training and Inference")
    parser.add_argument("--mode", choices=["train", "inference"], required=True, help="Mode: train or inference")
    parser.add_argument("--data_path", type=str, default="data/", help="Path to data directory")
    parser.add_argument("--model_path", type=str, default="models/", help="Path to save/load models")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--fold", type=int, default=None, help="Fold number for training (1-5)")
    
    args = parser.parse_args()
    
    config = Config()
    if args.config:
        config.load_config(args.config)
    
    pipeline = QuestionUnderstandingPipeline(config)
    
    if args.mode == "train":
        if args.fold:
            pipeline.train_fold(args.fold, args.data_path, args.model_path)
        else:
            pipeline.train_all_folds(args.data_path, args.model_path)
    elif args.mode == "inference":
        pipeline.inference(args.data_path, args.model_path)


if __name__ == "__main__":
    main()
