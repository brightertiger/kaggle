"""
RSNA Intracranial Hemorrhage Detection - Deep Learning Pipeline

A comprehensive deep learning solution for detecting intracranial hemorrhage 
in medical CT scans using state-of-the-art CNN architectures.
"""

from .core import Config
from .pipeline import IntracranialHemorrhagePipeline
from .data import (
    IntracranialHemorrhageDataset,
    create_data_loaders,
    create_inference_loader,
    preprocess_all_data,
    analyze_dataset
)
from .models import (
    create_model,
    WeightedBCELoss,
    FocalLoss,
    DiceLoss
)
from .training import (
    ModelTrainer,
    train_fold,
    create_optimizer,
    create_scheduler
)
from .inference import (
    ModelPredictor,
    ModelValidator,
    generate_all_predictions,
    run_validation
)

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@domain.com"

__all__ = [
    'Config',
    'IntracranialHemorrhagePipeline',
    'IntracranialHemorrhageDataset',
    'create_data_loaders',
    'create_inference_loader', 
    'preprocess_all_data',
    'analyze_dataset',
    'create_model',
    'WeightedBCELoss',
    'FocalLoss',
    'DiceLoss',
    'ModelTrainer',
    'train_fold',
    'create_optimizer',
    'create_scheduler',
    'ModelPredictor',
    'ModelValidator',
    'generate_all_predictions',
    'run_validation'
]
