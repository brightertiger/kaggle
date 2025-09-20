#!/usr/bin/env python3
"""
Whale Identification Challenge - Example Usage

This script demonstrates how to use the whale identification pipeline
with different approaches and configurations.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.pipeline import WhaleIdentificationPipeline

def example_basic_classification():
    """Example: Basic classification training"""
    print("=== Basic Classification Training ===")
    
    # Create configuration
    config = Config(
        data_dir="data",
        train_images_dir="data/train",
        test_images_dir="data/test",
        train_csv="data/train.csv",
        image_size=224,  # Smaller size for faster training
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=10,
        model_save_dir="models/basic_classification"
    )
    
    # Initialize pipeline
    pipeline = WhaleIdentificationPipeline(config)
    
    # Train classification model
    history = pipeline.train_classification_model(
        train_csv_path="data/train.csv",
        image_dir="data/train",
        model_name="basic_classification"
    )
    
    print("Basic classification training completed!")
    return pipeline

def example_center_loss_training():
    """Example: Training with center loss for better embeddings"""
    print("\n=== Center Loss Training ===")
    
    # Create configuration with center loss
    config = Config(
        data_dir="data",
        train_images_dir="data/train",
        test_images_dir="data/test",
        train_csv="data/train.csv",
        image_size=448,
        batch_size=64,
        learning_rate=1e-3,
        num_epochs=20,
        center_loss_weight=0.5,  # Weight for center loss
        model_save_dir="models/center_loss"
    )
    
    # Initialize pipeline
    pipeline = WhaleIdentificationPipeline(config)
    
    # Train with center loss
    history = pipeline.train_classification_model(
        train_csv_path="data/train.csv",
        image_dir="data/train",
        use_center_loss=True,
        model_name="center_loss"
    )
    
    print("Center loss training completed!")
    return pipeline

def example_pseudo_label_training():
    """Example: Training with pseudo labels"""
    print("\n=== Pseudo Label Training ===")
    
    # Create configuration
    config = Config(
        data_dir="data",
        train_images_dir="data/train",
        test_images_dir="data/test",
        train_csv="data/train.csv",
        image_size=448,
        batch_size=64,
        learning_rate=1e-3,
        num_epochs=15,
        model_save_dir="models/pseudo_label"
    )
    
    # Initialize pipeline
    pipeline = WhaleIdentificationPipeline(config)
    
    # Train with pseudo labels (if available)
    if os.path.exists("data/pseudo_labels.csv"):
        history = pipeline.train_with_pseudo_labels(
            train_csv_path="data/train.csv",
            pseudo_csv_path="data/pseudo_labels.csv",
            image_dir="data/train",
            model_name="pseudo_label"
        )
        print("Pseudo label training completed!")
    else:
        print("Pseudo labels file not found. Skipping pseudo label training.")
    
    return pipeline

def example_siamese_training():
    """Example: Training siamese network"""
    print("\n=== Siamese Network Training ===")
    
    # Create configuration
    config = Config(
        data_dir="data",
        train_images_dir="data/train",
        test_images_dir="data/test",
        train_csv="data/train.csv",
        image_size=448,
        batch_size=32,  # Smaller batch for siamese
        pair_model_lr=1e-4,
        pair_model_epochs=10,
        model_save_dir="models/siamese"
    )
    
    # Initialize pipeline
    pipeline = WhaleIdentificationPipeline(config)
    
    # Train siamese model (requires pretrained backbone)
    backbone_path = "models/center_loss/model.pth"
    if os.path.exists(backbone_path):
        history = pipeline.train_siamese_model(
            train_csv_path="data/train.csv",
            image_dir="data/train",
            backbone_path=backbone_path,
            model_name="siamese"
        )
        print("Siamese training completed!")
    else:
        print(f"Backbone model not found at {backbone_path}")
        print("Please train a classification model first.")
    
    return pipeline

def example_prediction():
    """Example: Generating predictions"""
    print("\n=== Generating Predictions ===")
    
    # Create configuration
    config = Config(
        data_dir="data",
        test_images_dir="data/test",
        image_size=448,
        batch_size=64,
        model_save_dir="models"
    )
    
    # Initialize pipeline
    pipeline = WhaleIdentificationPipeline(config)
    
    # Generate predictions
    model_path = "models/center_loss/model.pth"
    if os.path.exists(model_path):
        submission = pipeline.predict(
            test_image_dir="data/test",
            model_path=model_path,
            model_type="classification"
        )
        
        # Save predictions
        submission.to_csv("submission.csv", index=False)
        print(f"Predictions saved to submission.csv")
        print(f"Submission shape: {submission.shape}")
        print("\nSample predictions:")
        print(submission.head())
    else:
        print(f"Model not found at {model_path}")
        print("Please train a model first.")

def example_full_pipeline():
    """Example: Complete training pipeline"""
    print("\n=== Full Training Pipeline ===")
    
    # Create comprehensive configuration
    config = Config(
        data_dir="data",
        train_images_dir="data/train",
        test_images_dir="data/test",
        train_csv="data/train.csv",
        image_size=448,
        batch_size=64,
        learning_rate=1e-3,
        num_epochs=20,
        center_loss_weight=0.5,
        pair_model_lr=1e-4,
        pair_model_epochs=10,
        model_save_dir="models/full_pipeline"
    )
    
    # Initialize pipeline
    pipeline = WhaleIdentificationPipeline(config)
    
    # Step 1: Basic classification
    print("\n1. Training basic classification model...")
    pipeline.train_classification_model(
        train_csv_path="data/train.csv",
        image_dir="data/train",
        model_name="classification"
    )
    
    # Step 2: Center loss training
    print("\n2. Training with center loss...")
    pipeline.train_classification_model(
        train_csv_path="data/train.csv",
        image_dir="data/train",
        use_center_loss=True,
        model_name="center_loss"
    )
    
    # Step 3: Pseudo label training (if available)
    print("\n3. Training with pseudo labels...")
    if os.path.exists("data/pseudo_labels.csv"):
        pipeline.train_with_pseudo_labels(
            train_csv_path="data/train.csv",
            pseudo_csv_path="data/pseudo_labels.csv",
            image_dir="data/train",
            model_name="pseudo_label"
        )
    
    # Step 4: Siamese training
    print("\n4. Training siamese model...")
    backbone_path = "models/full_pipeline/center_loss/model.pth"
    if os.path.exists(backbone_path):
        pipeline.train_siamese_model(
            train_csv_path="data/train.csv",
            image_dir="data/train",
            backbone_path=backbone_path,
            model_name="siamese"
        )
    
    # Step 5: Generate predictions
    print("\n5. Generating predictions...")
    best_model_path = "models/full_pipeline/center_loss/model.pth"
    if os.path.exists(best_model_path):
        submission = pipeline.predict(
            test_image_dir="data/test",
            model_path=best_model_path,
            model_type="classification"
        )
        submission.to_csv("full_pipeline_submission.csv", index=False)
        print("Full pipeline completed!")
        print("Predictions saved to full_pipeline_submission.csv")
    
    return pipeline

def main():
    """Run example usage scenarios"""
    print("Whale Identification Challenge - Example Usage")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("Data directory not found!")
        print("Please ensure you have the following structure:")
        print("data/")
        print("├── train/")
        print("├── test/")
        print("├── train.csv")
        print("└── pseudo_labels.csv (optional)")
        return
    
    # Run examples
    try:
        # Basic examples
        example_basic_classification()
        example_center_loss_training()
        example_pseudo_label_training()
        example_siamese_training()
        example_prediction()
        
        # Full pipeline example
        example_full_pipeline()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Please check your data structure and file paths.")

if __name__ == "__main__":
    main()
