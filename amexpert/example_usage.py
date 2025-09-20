#!/usr/bin/env python3

"""
Example usage of the AmExpert Pipeline

This script demonstrates how to use the refactored AmExpert pipeline
for coupon redemption prediction.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import AmExpertPipeline


def main():
    print("AmExpert Coupon Redemption Prediction - Example Usage")
    print("=" * 60)
    
    # Initialize the pipeline
    pipeline = AmExpertPipeline(
        data_dir='data',
        feature_dir='data/feature',
        model_dir='data/model',
        score_dir='data/score'
    )
    
    print("\n1. Data Preprocessing")
    print("-" * 30)
    driver, train, valid, test = pipeline.preprocess_data()
    print(f"✓ Data loaded - Train: {train.shape}, Valid: {valid.shape}, Test: {test.shape}")
    
    print("\n2. Feature Engineering")
    print("-" * 30)
    pipeline.create_features()
    print("✓ All features created successfully")
    
    print("\n3. Feature Merging")
    print("-" * 30)
    train_final, valid_final, test_final = pipeline.merge_features()
    print(f"✓ Features merged - Final shapes: Train: {train_final.shape}, Valid: {valid_final.shape}")
    
    print("\n4. Model Training")
    print("-" * 30)
    pipeline.train_models()
    print("✓ All models trained successfully")
    
    print("\n5. Prediction Blending")
    print("-" * 30)
    final_predictions = pipeline.blend_predictions()
    print(f"✓ Final predictions ready - Shape: {final_predictions.shape}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("Final predictions saved to: score.csv")


if __name__ == '__main__':
    main()
