#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class Config:
    """Centralized configuration management for Jigsaw Toxic Comment Classification."""
    
    # Data paths
    data_path: str = "../data"
    model_path: str = "../model"
    output_path: str = "../output"
    
    # Model configuration
    random_seed: int = 42
    device: str = "cuda:0"
    
    # Data processing
    max_length: int = 222
    test_size: float = 0.2
    n_folds: int = 5
    
    # Identity columns for bias evaluation
    identity_columns: List[str] = field(default_factory=lambda: [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
    ])
    
    # Auxiliary labels
    aux_labels: List[str] = field(default_factory=lambda: [
        'target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat'
    ])
    
    # BERT configuration
    bert_config: Dict[str, Any] = field(default_factory=lambda: {
        'model_name': 'bert-base-uncased',
        'learning_rate': 2e-5,
        'batch_size': 12,
        'valid_batch_size': 8,
        'num_epochs': 3,
        'warmup_ratio': 0.05,
        'weight_decay': 0.01,
        'dropout': 0.1,
        'hidden_size': 768,
        'gradient_accumulation_steps': 5,
        'max_grad_norm': 1.0,
        'save_steps': 500,
        'eval_steps': 500,
        'logging_steps': 100
    })
    
    # GPT configuration
    gpt_config: Dict[str, Any] = field(default_factory=lambda: {
        'model_name': 'gpt2',
        'learning_rate': 2e-5,
        'batch_size': 12,
        'valid_batch_size': 5,
        'num_epochs': 3,
        'warmup_ratio': 0.05,
        'weight_decay': 0.01,
        'dropout': 0.1,
        'hidden_size': 768,
        'gradient_accumulation_steps': 5,
        'max_grad_norm': 1.0,
        'save_steps': 500,
        'eval_steps': 500,
        'logging_steps': 100
    })
    
    # Training configuration
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        'num_workers': 4,
        'pin_memory': True,
        'drop_last': True,
        'early_stopping_patience': 3,
        'save_best_model': True,
        'mixed_precision': False
    })
    
    # Evaluation configuration
    eval_config: Dict[str, Any] = field(default_factory=lambda: {
        'subgroup_auc_weight': -5,
        'overall_model_weight': 0.25,
        'toxicity_threshold': 0.5,
        'identity_threshold': 0.5
    })
    
    def get_data_path(self, subfolder: str = "", filename: str = "") -> str:
        """Get data file path."""
        if subfolder:
            path = os.path.join(self.data_path, subfolder)
        else:
            path = self.data_path
        
        if filename:
            path = os.path.join(path, filename)
        
        return path
    
    def get_model_path(self, subfolder: str = "", filename: str = "") -> str:
        """Get model file path."""
        if subfolder:
            path = os.path.join(self.model_path, subfolder)
        else:
            path = self.model_path
        
        if filename:
            path = os.path.join(path, filename)
        
        return path
    
    def get_output_path(self, subfolder: str = "", filename: str = "") -> str:
        """Get output file path."""
        if subfolder:
            path = os.path.join(self.output_path, subfolder)
        else:
            path = self.output_path
        
        if filename:
            path = os.path.join(path, filename)
        
        return path
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data_path,
            self.model_path,
            self.output_path,
            self.get_model_path('bert'),
            self.get_model_path('gpt'),
            self.get_output_path('predictions'),
            self.get_output_path('evaluations'),
            self.get_output_path('submissions')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def validate_setup(self) -> bool:
        """Validate that required files exist."""
        required_files = [
            self.get_data_path('', 'train.csv'),
            self.get_data_path('', 'test.csv')
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"❌ Required file not found: {file_path}")
                return False
        
        print("✅ Configuration validation passed")
        return True
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"⚠️ Unknown configuration key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data_path': self.data_path,
            'model_path': self.model_path,
            'output_path': self.output_path,
            'random_seed': self.random_seed,
            'device': self.device,
            'max_length': self.max_length,
            'test_size': self.test_size,
            'n_folds': self.n_folds,
            'identity_columns': self.identity_columns,
            'aux_labels': self.aux_labels,
            'bert_config': self.bert_config,
            'gpt_config': self.gpt_config,
            'training_config': self.training_config,
            'eval_config': self.eval_config
        }
