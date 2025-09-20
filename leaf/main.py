import argparse
import torch
from src.pipeline import CassavaPipeline
from src.utils.config import Config

def main():
    parser = argparse.ArgumentParser(description='Cassava Leaf Disease Classification')
    parser.add_argument('--mode', type=str, 
                       choices=['prepare_data', 'train', 'score', 'ensemble', 'full_pipeline'], 
                       default='train', help='Mode to run')
    parser.add_argument('--fold', type=int, default=0, help='Fold to train')
    parser.add_argument('--version', type=str, choices=['pretrain', 'version0', 'version1', 'version2', 'version3', 'version4', 'version5', 'version6', 'version7'], 
                       default='version7', help='Model version')
    parser.add_argument('--data_dir', type=str, default='../../data', help='Data directory path')
    parser.add_argument('--test_path', type=str, default='../../data/test.csv', help='Test data path')
    
    args = parser.parse_args()
    
    pipeline = CassavaPipeline()
    
    if args.mode == 'prepare_data':
        pipeline.prepare_data(args.data_dir)
    
    elif args.mode == 'train':
        pipeline.train_model(args.version, args.fold)
    
    elif args.mode == 'score':
        pipeline.score_model(args.version, args.test_path)
    
    elif args.mode == 'ensemble':
        pipeline.create_ensemble()
    
    elif args.mode == 'full_pipeline':
        pipeline.run_full_pipeline(args.data_dir, args.test_path)

if __name__ == '__main__':
    main()
