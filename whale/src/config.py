from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Config:
    data_dir: str = "data"
    train_images_dir: Optional[str] = None
    test_images_dir: Optional[str] = None
    train_csv: Optional[str] = None
    num_classes: int = 5004  # Inferred from the training vocabulary by the pipeline.
    embedding_dim: int = 256
    image_size: int = 448
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    num_epochs: int = 20
    freeze_layers: int = 1  # Retained legacy meaning: final backbone blocks left trainable.
    center_loss_weight: float = 0.5
    use_augmentation: bool = True
    use_weighted_sampling: bool = False
    model_save_dir: str = "models"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 0
    seed: int = 42
    pretrained: bool = True
    backbone_name: str = "resnet50"
    head_dim: int = 2048
    pseudo_epochs: int = 5
    pseudo_label_threshold: float = 0.9
    pair_model_lr: float = 1e-4
    pair_model_epochs: int = 10
    new_whale_threshold: Optional[float] = None

    def __post_init__(self):
        root = Path(self.data_dir)
        self.train_images_dir = self.train_images_dir or str(root / "train")
        self.test_images_dir = self.test_images_dir or str(root / "test")
        self.train_csv = self.train_csv or str(root / "train.csv")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for the BatchNorm training heads")
        if self.num_epochs < 1 or self.pair_model_epochs < 1 or self.pseudo_epochs < 1:
            raise ValueError("Training epochs must be positive")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")
        if self.new_whale_threshold is not None and not 0 <= self.new_whale_threshold <= 1:
            raise ValueError("new_whale_threshold must be between 0 and 1")
