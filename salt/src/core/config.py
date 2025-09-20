import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Configuration class for Salt Identification from Aerial Images project"""
    
    def __init__(self):
        # Data paths
        self.DATA_DIR = Path("../data")
        self.RAW_DATA_DIR = self.DATA_DIR / "download"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "data"
        self.MODEL_DIR = self.PROCESSED_DATA_DIR / "model"
        self.SCORES_DIR = self.PROCESSED_DATA_DIR / "scores"
        self.SUBMIT_DIR = self.PROCESSED_DATA_DIR / "submit"
        
        # Model configuration
        self.IMAGE_SIZE = 101
        self.PADDED_SIZE = 128
        self.BATCH_SIZE_TRAIN = 32
        self.BATCH_SIZE_VALID = 32
        self.NUM_EPOCHS = 200
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 0.0001
        self.DROPOUT = 0.25
        
        # Training configuration
        self.NUM_FOLDS = 5
        self.EARLY_STOPPING_PATIENCE = 50
        self.LR_REDUCTION_PATIENCE = 10
        self.LR_REDUCTION_FACTOR = 0.5
        
        # Model architecture
        self.PRETRAINED = True
        self.MODEL_NAME = "seresnet34"
        
        # Data augmentation
        self.USE_FLIP_AUGMENTATION = True
        self.FLIP_PROBABILITY = 0.5
        
        # Device configuration
        self.DEVICE = self._get_device()
        
        # Random seeds
        self.RANDOM_SEED = 2017
        
        # Image preprocessing
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        
        # Loss function configuration
        self.LOSS_TYPE = "lovasz"  # Options: "dice", "lovasz", "bce"
        self.DICE_WEIGHT = 1.0
        self.BCE_WEIGHT = 1.0
        
        # Evaluation metrics
        self.METRIC_TYPE = "iou"
        self.IOU_CUTOFF = -0.18
        self.IOU_SQUASH = False
        
        # Submission configuration
        self.MIN_SALT_PIXELS = 25
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _get_device(self) -> str:
        """Automatically detect and set the best available device"""
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        else:
            return "cpu"
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODEL_DIR,
            self.SCORES_DIR / "valid",
            self.SCORES_DIR / "test",
            self.SUBMIT_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def print_config(self):
        """Print current configuration"""
        print("Salt Segmentation Configuration:")
        print("=" * 40)
        for key, value in self.to_dict().items():
            print(f"{key}: {value}")
        print("=" * 40)
