import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.pipeline import PronounResolutionPipeline

def main():
    parser = argparse.ArgumentParser(description='Pronoun Resolution Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, help='Mode to run')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    config = Config(args.config)
    pipeline = PronounResolutionPipeline(config, args.device)
    
    if args.mode == 'train':
        pipeline.train()
    elif args.mode == 'predict':
        pipeline.predict()

if __name__ == '__main__':
    main()
