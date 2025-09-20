import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Data paths
    data_dir: str = "data"
    train_images_dir: str = "data/train"
    test_images_dir: str = "data/test"
    train_csv: str = "data/train.csv"
    
    # Model parameters
    num_classes: int = 5004
    embedding_dim: int = 256
    image_size: int = 448
    batch_size: int = 64
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    num_epochs: int = 20
    freeze_layers: int = 1
    
    # Center loss parameters
    center_loss_weight: float = 0.5
    
    # Data augmentation
    use_augmentation: bool = True
    use_weighted_sampling: bool = False
    
    # Model saving
    model_save_dir: str = "models"
    checkpoint_frequency: int = 5
    
    # Device
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Validation split
    validation_split: float = 0.2
    
    # Pseudo labeling
    use_pseudo_labels: bool = True
    pseudo_label_threshold: float = 0.9
    
    # Pair model parameters
    pair_model_lr: float = 1e-4
    pair_model_epochs: int = 10
