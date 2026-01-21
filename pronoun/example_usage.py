import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.pipeline import PronounResolutionPipeline

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config = Config.default()
    
    pipeline = PronounResolutionPipeline(config, device)
    
    print("Starting training...")
    pipeline.train()
    
    print("Generating predictions...")
    predictions = pipeline.predict()
    print(f"Generated predictions for {len(predictions)} samples")
    print(predictions.head())

if __name__ == '__main__':
    main()
