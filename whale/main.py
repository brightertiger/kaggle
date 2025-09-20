#!/usr/bin/env python3
"""
Whale Identification Challenge - Main Entry Point

This script provides a command-line interface for training and evaluating
whale identification models using various approaches including classification,
pseudo-labeling, center loss, and siamese networks.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.pipeline import WhaleIdentificationPipeline
from src.data_utils import create_data_loaders, create_test_loader

def main():
    parser = argparse.ArgumentParser(description="Whale Identification Challenge")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train_classification", "train_pseudo", "train_center_loss", 
                               "train_siamese", "predict", "full_pipeline"],
                       help="Mode to run the pipeline")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing the dataset")
    parser.add_argument("--train_csv", type=str, default="data/train.csv",
                       help="Path to training CSV file")
    parser.add_argument("--test_csv", type=str, default="data/test.csv",
                       help="Path to test CSV file")
    parser.add_argument("--pseudo_csv", type=str, default="data/pseudo_labels.csv",
                       help="Path to pseudo labels CSV file")
    parser.add_argument("--image_size", type=int, default=448,
                       help="Image size for training")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--model_dir", type=str, default="models",
                       help="Directory to save models")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model checkpoint for inference")
    parser.add_argument("--output_file", type=str, default="submission.csv",
                       help="Output file for predictions")
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        data_dir=args.data_dir,
        train_images_dir=os.path.join(args.data_dir, "train"),
        test_images_dir=os.path.join(args.data_dir, "test"),
        train_csv=args.train_csv,
        image_size=args.image_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        model_save_dir=args.model_dir
    )
    
    # Initialize pipeline
    pipeline = WhaleIdentificationPipeline(config)
    
    if args.mode == "train_classification":
        print("Training classification model...")
        history = pipeline.train_classification_model(
            train_csv_path=args.train_csv,
            image_dir=config.train_images_dir,
            model_name="classification"
        )
        print("Training completed!")
        
    elif args.mode == "train_pseudo":
        print("Training with pseudo labels...")
        history = pipeline.train_with_pseudo_labels(
            train_csv_path=args.train_csv,
            pseudo_csv_path=args.pseudo_csv,
            image_dir=config.train_images_dir,
            model_name="pseudo_label"
        )
        print("Training completed!")
        
    elif args.mode == "train_center_loss":
        print("Training with center loss...")
        history = pipeline.train_classification_model(
            train_csv_path=args.train_csv,
            image_dir=config.train_images_dir,
            use_center_loss=True,
            model_name="center_loss"
        )
        print("Training completed!")
        
    elif args.mode == "train_siamese":
        print("Training siamese model...")
        # Need a pretrained backbone
        backbone_path = os.path.join(args.model_dir, "classification", "model.pth")
        if not os.path.exists(backbone_path):
            print(f"Backbone model not found at {backbone_path}")
            print("Please train a classification model first.")
            return
        
        history = pipeline.train_siamese_model(
            train_csv_path=args.train_csv,
            image_dir=config.train_images_dir,
            backbone_path=backbone_path,
            model_name="siamese"
        )
        print("Training completed!")
        
    elif args.mode == "predict":
        print("Generating predictions...")
        if args.model_path is None:
            print("Please specify --model_path for inference")
            return
            
        submission = pipeline.predict(
            test_image_dir=config.test_images_dir,
            model_path=args.model_path,
            model_type="classification"
        )
        
        submission.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")
        
    elif args.mode == "full_pipeline":
        print("Running full training pipeline...")
        
        # 1. Train classification model
        print("\n1. Training classification model...")
        pipeline.train_classification_model(
            train_csv_path=args.train_csv,
            image_dir=config.train_images_dir,
            model_name="classification"
        )
        
        # 2. Train with pseudo labels
        print("\n2. Training with pseudo labels...")
        if os.path.exists(args.pseudo_csv):
            pipeline.train_with_pseudo_labels(
                train_csv_path=args.train_csv,
                pseudo_csv_path=args.pseudo_csv,
                image_dir=config.train_images_dir,
                model_name="pseudo_label"
            )
        
        # 3. Train with center loss
        print("\n3. Training with center loss...")
        pipeline.train_classification_model(
            train_csv_path=args.train_csv,
            image_dir=config.train_images_dir,
            use_center_loss=True,
            model_name="center_loss"
        )
        
        # 4. Train siamese model
        print("\n4. Training siamese model...")
        backbone_path = os.path.join(args.model_dir, "classification", "model.pth")
        if os.path.exists(backbone_path):
            pipeline.train_siamese_model(
                train_csv_path=args.train_csv,
                image_dir=config.train_images_dir,
                backbone_path=backbone_path,
                model_name="siamese"
            )
        
        # 5. Generate predictions
        print("\n5. Generating predictions...")
        best_model_path = os.path.join(args.model_dir, "center_loss", "model.pth")
        if os.path.exists(best_model_path):
            submission = pipeline.predict(
                test_image_dir=config.test_images_dir,
                model_path=best_model_path,
                model_type="classification"
            )
            submission.to_csv(args.output_file, index=False)
            print(f"Predictions saved to {args.output_file}")
        
        print("Full pipeline completed!")
    
    # Save training history
    history_file = os.path.join(args.model_dir, "training_history.json")
    pipeline.save_training_history(history_file)
    print(f"Training history saved to {history_file}")

if __name__ == "__main__":
    main()
