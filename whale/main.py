#!/usr/bin/env python3
"""
Whale Identification Challenge - Main Entry Point

This script provides a command-line interface for training and evaluating
whale identification models using various approaches including classification,
pseudo-labeling, center loss, and siamese networks.
"""

import argparse
import os
from pathlib import Path

from src.config import Config
from src.pipeline import WhaleIdentificationPipeline

def main():
    parser = argparse.ArgumentParser(description="Whale Identification Challenge")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train_classification", "train_pseudo", "train_center_loss", 
                               "train_siamese", "predict", "full_pipeline"],
                       help="Mode to run the pipeline")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing the dataset")
    parser.add_argument("--train_csv", type=str, default=None,
                       help="Path to training CSV file")
    parser.add_argument("--test_csv", type=str, default=None,
                       help="Optional sample_submission.csv to preserve test image order")
    parser.add_argument("--pseudo_csv", type=str, default=None,
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
    
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_pretrained", action="store_true", help="Random initialization; no weight downloads")
    parser.add_argument("--backbone", choices=["resnet50", "resnet18"], default="resnet50")
    parser.add_argument("--head_dim", type=int, default=2048)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--pair_epochs", type=int, default=10)
    parser.add_argument("--pseudo_epochs", type=int, default=5)
    parser.add_argument("--pseudo_image_dir", default=None)
    parser.add_argument("--val_csv", default=None, help="Explicit validation CSV; filenames must be disjoint from train_csv")
    parser.add_argument("--model_type", choices=["classification", "siamese"], default="classification")
    parser.add_argument("--new_whale_threshold", type=float, default=None,
                        help="Insert new_whale where known-identity scores fall below this threshold")
    args = parser.parse_args()
    args.train_csv = args.train_csv or str(Path(args.data_dir) / "train.csv")
    args.pseudo_csv = args.pseudo_csv or str(Path(args.data_dir) / "pseudo_labels.csv")
    
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
        model_save_dir=args.model_dir,
        device=args.device or Config().device,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
        backbone_name=args.backbone,
        head_dim=args.head_dim,
        embedding_dim=args.embedding_dim,
        pair_model_epochs=args.pair_epochs,
        pseudo_epochs=args.pseudo_epochs,
        new_whale_threshold=args.new_whale_threshold
    )
    
    # Initialize pipeline
    pipeline = WhaleIdentificationPipeline(config)
    
    if args.mode == "train_classification":
        print("Training classification model...")
        history = pipeline.train_classification_model(
            train_csv_path=args.train_csv,
            image_dir=config.train_images_dir,
            model_name="classification", val_csv_path=args.val_csv
        )
        print("Training completed!")
        
    elif args.mode == "train_pseudo":
        print("Training with pseudo labels...")
        history = pipeline.train_with_pseudo_labels(
            train_csv_path=args.train_csv,
            pseudo_csv_path=args.pseudo_csv,
            image_dir=config.train_images_dir,
            model_name="pseudo_label", pseudo_image_dir=args.pseudo_image_dir, val_csv_path=args.val_csv
        )
        print("Training completed!")
        
    elif args.mode == "train_center_loss":
        print("Training with center loss...")
        history = pipeline.train_classification_model(
            train_csv_path=args.train_csv,
            image_dir=config.train_images_dir,
            use_center_loss=True,
            model_name="center_loss", val_csv_path=args.val_csv
        )
        print("Training completed!")
        
    elif args.mode == "train_siamese":
        print("Training siamese model...")
        # Need a pretrained backbone
        backbone_path = args.model_path or os.path.join(args.model_dir, "center_loss", "model.pth")
        if not os.path.exists(backbone_path):
            parser.error(f"Backbone not found: {backbone_path}; train center loss or pass --model_path")
        
        history = pipeline.train_siamese_model(
            train_csv_path=args.train_csv,
            image_dir=config.train_images_dir,
            backbone_path=backbone_path,
            model_name="siamese", val_csv_path=args.val_csv
        )
        print("Training completed!")
        
    elif args.mode == "predict":
        print("Generating predictions...")
        if args.model_path is None:
            parser.error("Please specify --model_path for inference")
            
        submission = pipeline.predict(
            test_image_dir=config.test_images_dir,
            model_path=args.model_path,
            model_type=args.model_type, test_csv_path=args.test_csv
        )
        
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")
        
    elif args.mode == "full_pipeline":
        print("Running full training pipeline...")
        
        # 1. Train classification model
        print("\n1. Training classification model...")
        pipeline.train_classification_model(
            train_csv_path=args.train_csv,
            image_dir=config.train_images_dir,
            model_name="classification", val_csv_path=args.val_csv
        )
        
        # 2. Train with pseudo labels
        print("\n2. Training with pseudo labels...")
        if os.path.exists(args.pseudo_csv):
            pipeline.train_with_pseudo_labels(
                train_csv_path=args.train_csv,
                pseudo_csv_path=args.pseudo_csv,
                image_dir=config.train_images_dir,
                model_name="pseudo_label", pseudo_image_dir=args.pseudo_image_dir, val_csv_path=args.val_csv
            )
        
        else:
            print("No pseudo-label CSV found; skipping optional pseudo-label pretraining.")

        # 3. Train with center loss
        print("\n3. Training with center loss...")
        pipeline.train_classification_model(
            train_csv_path=args.train_csv,
            image_dir=config.train_images_dir,
            use_center_loss=True,
            model_name="center_loss", val_csv_path=args.val_csv
        )
        
        # 4. Train the pair head on the center-loss embedding.
        backbone_path = os.path.join(args.model_dir, "center_loss", "model.pth")
        pipeline.train_siamese_model(
            train_csv_path=args.train_csv, image_dir=config.train_images_dir,
            backbone_path=backbone_path, model_name="siamese", val_csv_path=args.val_csv
        )

        # 5. The full pipeline submission uses the center-loss classifier.
        submission = pipeline.predict(
            test_image_dir=config.test_images_dir, model_path=backbone_path,
            model_type="classification", test_csv_path=args.test_csv
        )
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")

        print("Full pipeline completed!")
    
    # Save training history
    if args.mode != "predict":
        history_file = os.path.join(args.model_dir, "training_history.json")
        pipeline.save_training_history(history_file)
        print(f"Training history saved to {history_file}")

if __name__ == "__main__":
    main()
