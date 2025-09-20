#!/usr/bin/env python3

"""
Jigsaw Toxic Comment Classification Package

A comprehensive solution for toxic comment classification with bias-aware training
and evaluation using BERT and GPT-2 models.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import Config
from .data_utils import DataProcessor, DataLoader
from .models import BERTClassifier, GPTClassifier, ModelTrainer, CustomLoss, ToxicCommentDataset
from .evaluation import BiasEvaluator, ModelEvaluator
from .pipeline import JigsawPipeline

__all__ = [
    'Config',
    'DataProcessor',
    'DataLoader', 
    'BERTClassifier',
    'GPTClassifier',
    'ModelTrainer',
    'CustomLoss',
    'ToxicCommentDataset',
    'BiasEvaluator',
    'ModelEvaluator',
    'JigsawPipeline'
]
