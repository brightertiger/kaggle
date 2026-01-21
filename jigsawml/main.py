import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import JigsawPipeline
from src.utils.config import Config

def main():
    parser = argparse.ArgumentParser(description='Jigsaw Multilingual Toxic Comment Classification')
    parser.add_argument('--mode', type=str, 
                       choices=['prepare_data', 'embeddings', 'adversarial', 'train', 'score', 'ensemble', 'full_pipeline'], 
                       default='train', help='Mode to run')
    parser.add_argument('--subset', type=int, default=0, help='Subset/fold to train')
    parser.add_argument('--version', type=int, choices=[1, 2], default=1, help='Model version')
    parser.add_argument('--load_pretrained', action='store_true', help='Load pretrained weights')
    parser.add_argument('--data_dir', type=str, default='../../data', help='Data directory path')
    parser.add_argument('--test_path', type=str, default='../../data/process/foreign/test_english.csv', help='Test data path')
    
    args = parser.parse_args()
    
    pipeline = JigsawPipeline()
    
    if args.mode == 'prepare_data':
        pipeline.prepare_data(args.data_dir)
    
    elif args.mode == 'embeddings':
        pipeline.generate_embeddings(args.data_dir)
    
    elif args.mode == 'adversarial':
        pipeline.generate_adversarial_data(args.data_dir)
    
    elif args.mode == 'train':
        if args.version == 1:
            pipeline.train_version1(args.subset, args.load_pretrained)
        else:
            pipeline.train_version2(args.subset, args.load_pretrained)
    
    elif args.mode == 'score':
        pipeline.scoring_pipeline.score_all_models(args.test_path)
    
    elif args.mode == 'ensemble':
        pipeline.create_ensemble()
    
    elif args.mode == 'full_pipeline':
        pipeline.run_full_pipeline(args.data_dir, args.test_path)

if __name__ == '__main__':
    main()
