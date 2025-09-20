"""
APTOS Diabetic Retinopathy Detection Package

This package contains all the modules for training deep learning models
to detect diabetic retinopathy severity levels from retinal fundus images.
"""

try:
    from .config import Config
    from .data_utils import (
        DiabeticRetinopathyDataset, 
        NoiseAugmentedDataset, 
        ImageTransforms,
        create_data_loaders
    )
    from .model import DiabeticRetinopathyModel
    from .loss import DiabeticRetinopathyLoss, NoiseAugmentedLoss
    from .trainer import DiabeticRetinopathyTrainer, NoiseAugmentedTrainer
    from .optimizer import RAdam
    from .pipeline import APTOSPipeline
except ImportError as e:
    # Handle case where dependencies are not installed
    print(f"Warning: Some dependencies may not be installed: {e}")

__version__ = "1.0.0"
__author__ = "Data Science Portfolio"
