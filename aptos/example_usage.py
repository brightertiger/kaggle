"""
Example usage of the APTOS Diabetic Retinopathy Detection system.

This script demonstrates how to:
1. Load and preprocess data
2. Create and train a model
3. Make predictions on new images
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.data_utils import DiabeticRetinopathyDataset, ImageTransforms
from src.model import DiabeticRetinopathyModel
from src.trainer import DiabeticRetinopathyTrainer
from src.loss import DiabeticRetinopathyLoss
from src.optimizer import RAdam

def demonstrate_data_loading():
    """Demonstrate data loading and preprocessing."""
    print("=== Data Loading Demo ===")
    
    config = Config()
    
    # Create a sample dataset
    sample_data = {
        'id_code': ['sample_image_1', 'sample_image_2'],
        'diagnosis': [1, 3]
    }
    
    import pandas as pd
    df = pd.DataFrame(sample_data)
    
    # Create dataset
    dataset = DiabeticRetinopathyDataset(
        image_path="../../data/train/train",
        data=df,
        size=config.IMAGE_SIZE,
        weight=1.0,
        noise=False,
        config=config
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample item keys: {list(dataset[0].keys())}")
    print(f"Image shape: {dataset[0]['image'].shape}")
    print(f"Label: {dataset[0]['label'].item()}")

def demonstrate_model_creation():
    """Demonstrate model creation and architecture."""
    print("\n=== Model Creation Demo ===")
    
    config = Config()
    model = DiabeticRetinopathyModel(config.MODEL_NAME, config)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    # Forward pass
    with torch.no_grad():
        regression_output, classification_output = model(dummy_input)
    
    print(f"Model: {config.MODEL_NAME}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Regression output shape: {regression_output.shape}")
    print(f"Classification output shape: {classification_output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

def demonstrate_training_setup():
    """Demonstrate training setup and configuration."""
    print("\n=== Training Setup Demo ===")
    
    config = Config()
    
    # Create model
    model = DiabeticRetinopathyModel(config.MODEL_NAME, config)
    
    # Create optimizer
    optimizer = RAdam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create loss function
    loss_fn = DiabeticRetinopathyLoss(
        mse_weight=config.MSE_WEIGHT,
        variance_weight=0.0,
        config=config
    )
    
    # Create trainer
    trainer = DiabeticRetinopathyTrainer(config)
    
    print(f"Optimizer: RAdam")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Weight decay: {config.WEIGHT_DECAY}")
    print(f"MSE weight: {config.MSE_WEIGHT}")
    print(f"Device: {config.DEVICE}")

def demonstrate_prediction():
    """Demonstrate making predictions on new images."""
    print("\n=== Prediction Demo ===")
    
    config = Config()
    model = DiabeticRetinopathyModel(config.MODEL_NAME, config)
    
    # Create dummy image (simulating loaded image)
    dummy_image = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        regression_output, classification_output = model(dummy_image)
        
        # Convert to severity level
        predicted_severity = torch.round(regression_output.clamp(0, 4))
        predicted_probabilities = torch.softmax(classification_output, dim=1)
    
    print(f"Regression output: {regression_output.item():.3f}")
    print(f"Predicted severity: {predicted_severity.item()}")
    print(f"Classification probabilities: {predicted_probabilities.squeeze().tolist()}")
    
    # Map severity to description
    severity_map = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR", 
        3: "Severe DR",
        4: "Proliferative DR"
    }
    
    predicted_class = predicted_severity.item()
    print(f"Predicted condition: {severity_map[predicted_class]}")

def main():
    """Run all demonstrations."""
    print("APTOS Diabetic Retinopathy Detection - Example Usage")
    print("=" * 60)
    
    try:
        demonstrate_data_loading()
        demonstrate_model_creation()
        demonstrate_training_setup()
        demonstrate_prediction()
        
        print("\n=== Summary ===")
        print("✅ All demonstrations completed successfully!")
        print("📝 This example shows the basic usage of the refactored codebase.")
        print("🚀 For full training, run: python train.py")
        print("📊 For data preprocessing, run: python preprocess.py")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
