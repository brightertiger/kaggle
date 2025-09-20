"""
TalkingData AdTracking Fraud Detection

A comprehensive machine learning solution for detecting fraudulent ad clicks in mobile advertising,
developed for the TalkingData AdTracking Fraud Detection Challenge on Kaggle.

This package provides a complete pipeline for:
- Data preprocessing and feature engineering
- Multiple LightGBM model training
- Model ensemble and blending
- Click fraud prediction
"""

__version__ = "1.0.0"
__author__ = "Ujjwal Sharma"

from .core.config import Config
from .pipeline import TalkingDataPipeline

__all__ = ['Config', 'TalkingDataPipeline']
