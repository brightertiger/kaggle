#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.config import get_config
from src.pipeline import TweetSentimentPipeline

def main():
    print("Tweet Sentiment Analysis - Example Usage")
    print("=" * 50)
    
    config = get_config()
    
    print("Configuration loaded:")
    print(f"- Max sequence length: {config.data.max_length}")
    print(f"- Batch size: {config.data.batch_size}")
    print(f"- Number of folds: {config.data.n_folds}")
    print(f"- Learning rate: {config.model.learning_rate}")
    print(f"- Max epochs: {config.model.max_epochs}")
    
    pipeline = TweetSentimentPipeline(config)
    
    print("\nCreating sample data for demonstration...")
    
    sample_data = pd.DataFrame({
        'textID': ['1', '2', '3'],
        'text': [
            'I love this product!',
            'This is terrible.',
            'It\'s okay, nothing special.'
        ],
        'sentiment': ['positive', 'negative', 'neutral'],
        'selected_text': [
            'love this product',
            'terrible',
            'okay'
        ]
    })
    
    sample_path = 'data/raw/sample_train.csv'
    os.makedirs('data/raw', exist_ok=True)
    sample_data.to_csv(sample_path, index=False)
    
    print(f"Sample data created at: {sample_path}")
    print("\nSample data preview:")
    print(sample_data.head())
    
    print("\nNote: This is a demonstration. For actual training, you would need:")
    print("1. The full competition dataset")
    print("2. Pre-trained RoBERTa model files")
    print("3. Sufficient computational resources")
    
    print("\nTo run the full pipeline:")
    print("python main.py --mode train --data-path data/raw/train.csv")
    print("python main.py --mode evaluate --data-path data/raw/train.csv")
    print("python main.py --mode predict --test-path data/raw/test.csv")

if __name__ == '__main__':
    main()
