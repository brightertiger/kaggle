import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import os
import json

from .config import Config
from .models import WhaleResNet, CenterLoss, SiameseNetwork, Accuracy, BinaryAccuracy
from .data_utils import create_data_loaders, create_test_loader, create_pseudo_label_loader
from .trainer import Trainer, SiameseTrainer, ModelCheckpoint
from .optimizer import AdamW
from .scheduler import CosineLR
from tqdm import tqdm

class WhaleIdentificationPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.classification_model = None
        self.center_loss = None
        self.siamese_model = None
        
        # Training history
        self.training_history = {}
    
    def train_classification_model(self, train_csv_path: str, image_dir: str,
                                 val_csv_path: Optional[str] = None,
                                 use_center_loss: bool = False,
                                 model_name: str = "classification") -> Dict[str, Any]:
        """Train the main classification model"""
        
        print(f"Training {model_name} model...")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            self.config, train_csv_path, image_dir, return_valid=True
        )
        
        # Initialize model
        self.classification_model = WhaleResNet(
            num_classes=self.config.num_classes,
            freeze_layers=self.config.freeze_layers
        )
        
        # Initialize center loss if needed
        if use_center_loss:
            self.center_loss = CenterLoss(
                num_classes=self.config.num_classes,
                feat_dim=self.config.embedding_dim,
                use_gpu=self.device.type == 'cuda'
            )
        
        # Initialize optimizer
        optimizer = AdamW(
            self.classification_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize scheduler
        scheduler = CosineLR(
            optimizer,
            T_max=self.config.num_epochs,
            T_mult=0.98,
            eta_min=self.config.learning_rate * 0.01
        )
        
        # Initialize loss and metric functions
        loss_fn = nn.CrossEntropyLoss()
        metric_fn = Accuracy(topk=5)
        
        # Initialize trainer
        trainer = Trainer(self.classification_model, str(self.device))
        
        # Setup checkpointing
        checkpoint_dir = os.path.join(self.config.model_save_dir, model_name)
        checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss', mode='min')
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
            num_epochs=self.config.num_epochs,
            checkpoint=checkpoint,
            scheduler=scheduler,
            center_loss=self.center_loss,
            center_loss_weight=self.config.center_loss_weight if use_center_loss else 0.0
        )
        
        self.training_history[model_name] = history
        return history
    
    def train_with_pseudo_labels(self, train_csv_path: str, pseudo_csv_path: str,
                                image_dir: str, model_name: str = "pseudo_label") -> Dict[str, Any]:
        """Train model using pseudo labels from external data"""
        
        print(f"Training {model_name} model with pseudo labels...")
        
        # First train on pseudo labels
        pseudo_loader = create_pseudo_label_loader(self.config, pseudo_csv_path, image_dir)
        
        # Initialize model
        self.classification_model = WhaleResNet(
            num_classes=self.config.num_classes,
            freeze_layers=self.config.freeze_layers
        )
        
        # Initialize optimizer
        optimizer = AdamW(
            self.classification_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize scheduler
        scheduler = CosineLR(
            optimizer,
            T_max=self.config.num_epochs,
            T_mult=0.98,
            eta_min=self.config.learning_rate * 0.01
        )
        
        # Initialize loss and metric functions
        loss_fn = nn.CrossEntropyLoss()
        metric_fn = Accuracy(topk=5)
        
        # Initialize trainer
        trainer = Trainer(self.classification_model, str(self.device))
        
        # Train on pseudo labels first
        print("Training on pseudo labels...")
        pseudo_history = trainer.train(
            train_loader=pseudo_loader,
            val_loader=None,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
            num_epochs=5,  # Fewer epochs for pseudo labels
            checkpoint=None,
            scheduler=scheduler
        )
        
        # Then train on real data
        train_loader, val_loader = create_data_loaders(
            self.config, train_csv_path, image_dir, return_valid=True
        )
        
        print("Fine-tuning on real data...")
        real_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
            num_epochs=self.config.num_epochs,
            checkpoint=ModelCheckpoint(
                os.path.join(self.config.model_save_dir, model_name),
                monitor='val_loss', mode='min'
            ),
            scheduler=scheduler
        )
        
        # Combine histories
        combined_history = {
            'pseudo': pseudo_history,
            'real': real_history
        }
        
        self.training_history[model_name] = combined_history
        return combined_history
    
    def train_siamese_model(self, train_csv_path: str, image_dir: str,
                           backbone_path: str, model_name: str = "siamese") -> Dict[str, Any]:
        """Train siamese network for similarity learning"""
        
        print(f"Training {model_name} model...")
        
        # Create pair data loader (this would need to be implemented)
        # For now, we'll use a simplified approach
        train_loader, val_loader = create_data_loaders(
            self.config, train_csv_path, image_dir, return_valid=True
        )
        
        # Initialize siamese model
        self.siamese_model = SiameseNetwork(
            backbone_path=backbone_path,
            freeze_backbone=True,
            resnet_layers=None
        )
        
        # Initialize optimizer
        optimizer = AdamW(
            self.siamese_model.parameters(),
            lr=self.config.pair_model_lr,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize scheduler
        scheduler = CosineLR(
            optimizer,
            T_max=self.config.pair_model_epochs,
            T_mult=0.98,
            eta_min=self.config.pair_model_lr * 0.01
        )
        
        # Initialize loss and metric functions
        loss_fn = nn.BCELoss()
        metric_fn = BinaryAccuracy()
        
        # Initialize trainer
        trainer = SiameseTrainer(self.siamese_model, str(self.device))
        
        # Setup checkpointing
        checkpoint_dir = os.path.join(self.config.model_save_dir, model_name)
        checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss', mode='min')
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
            num_epochs=self.config.pair_model_epochs,
            checkpoint=checkpoint,
            scheduler=scheduler
        )
        
        self.training_history[model_name] = history
        return history
    
    def predict(self, test_image_dir: str, model_path: Optional[str] = None,
               model_type: str = "classification") -> pd.DataFrame:
        """Generate predictions on test data"""
        
        print("Generating predictions...")
        
        # Load model
        if model_path:
            if model_type == "classification":
                model = WhaleResNet(num_classes=self.config.num_classes)
            elif model_type == "siamese":
                model = SiameseNetwork(backbone_path=model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = self.classification_model if model_type == "classification" else self.siamese_model
        
        model.to(self.device)
        model.eval()
        
        # Create test loader
        test_loader = create_test_loader(self.config, test_image_dir)
        
        # Generate predictions
        predictions = []
        image_names = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                images = batch['image'].to(self.device)
                batch_names = batch['image_name']
                
                if model_type == "classification":
                    preds, _ = model(images)
                    probs = torch.softmax(preds, dim=1)
                    top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
                    
                    for i, name in enumerate(batch_names):
                        pred_string = " ".join([
                            f"{idx.item()}_{prob.item():.4f}" 
                            for idx, prob in zip(top5_indices[i], top5_probs[i])
                        ])
                        predictions.append(pred_string)
                        image_names.append(name)
                
                elif model_type == "siamese":
                    # For siamese model, we'd need pairs of images
                    # This is a simplified version
                    pass
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'Image': image_names,
            'Id': predictions
        })
        
        return submission
    
    def save_training_history(self, filepath: str):
        """Save training history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_training_history(self, filepath: str):
        """Load training history from JSON file"""
        with open(filepath, 'r') as f:
            self.training_history = json.load(f)
