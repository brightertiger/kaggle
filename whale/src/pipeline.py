"""Classification, pseudo-label fine-tuning, and pairwise whale identification."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .config import Config
from .data_utils import (create_data_loaders, create_test_loader, create_pseudo_label_loader,
                         create_pair_loaders, read_labels, WhaleDataset, _loader)
from .models import WhaleResNet, CenterLoss, SiameseNetwork, MeanAveragePrecision, BinaryAccuracy
from .trainer import Trainer, SiameseTrainer, ModelCheckpoint
from .optimizer import AdamW
from .scheduler import CosineLR


class WhaleIdentificationPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.classification_model = None
        self.center_loss = None
        self.siamese_model = None
        self.class_names = None
        self.training_history = {}

    def _new_classifier(self, class_names):
        self.class_names = list(class_names)
        self.config.num_classes = len(class_names)
        model = WhaleResNet(num_classes=len(class_names), freeze_layers=self.config.freeze_layers,
                            embedding_dim=self.config.embedding_dim, pretrained=self.config.pretrained,
                            backbone_name=self.config.backbone_name, head_dim=self.config.head_dim)
        model.class_names = self.class_names
        model.image_size = self.config.image_size
        return model.to(self.device)

    def _optimizer(self, model, lr, epochs, center_loss=None):
        params = [p for p in model.parameters() if p.requires_grad]
        if center_loss is not None:
            params.extend(center_loss.parameters())
        optimizer = AdamW(params, lr=lr, weight_decay=self.config.weight_decay)
        scheduler = CosineLR(optimizer, T_max=epochs, T_mult=0.98, eta_min=lr * 0.01)
        return optimizer, scheduler

    def _checkpoint(self, model_name):
        return ModelCheckpoint(str(Path(self.config.model_save_dir) / model_name))

    def train_classification_model(self, train_csv_path, image_dir, val_csv_path=None,
                                   use_center_loss=False, model_name="classification"):
        print(f"Training {model_name} model...")
        train_loader, val_loader = create_data_loaders(
            self.config, train_csv_path, image_dir, val_csv_path=val_csv_path)
        self.classification_model = self._new_classifier(train_loader.dataset.class_names)
        self.center_loss = None
        if use_center_loss:
            self.center_loss = CenterLoss(len(self.class_names), self.config.embedding_dim,
                                          use_gpu=False).to(self.device)
        optimizer, scheduler = self._optimizer(self.classification_model, self.config.learning_rate,
                                               self.config.num_epochs, self.center_loss)
        checkpoint = self._checkpoint(model_name)
        checkpoint.center_loss = self.center_loss
        history = Trainer(self.classification_model, str(self.device)).train(
            train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), MeanAveragePrecision(),
            self.config.num_epochs, checkpoint=checkpoint, scheduler=scheduler,
            center_loss=self.center_loss,
            center_loss_weight=self.config.center_loss_weight if use_center_loss else 0.0)
        self.training_history[model_name] = history
        return history

    def train_with_pseudo_labels(self, train_csv_path, pseudo_csv_path, image_dir,
                                 model_name="pseudo_label", pseudo_image_dir=None, val_csv_path=None):
        print(f"Training {model_name} model with pseudo labels...")
        train_loader, val_loader = create_data_loaders(
            self.config, train_csv_path, image_dir, val_csv_path=val_csv_path)
        self.classification_model = self._new_classifier(train_loader.dataset.class_names)
        # Never pretrain on the held-out competition images through the pseudo CSV.
        excluded = val_loader.dataset.images if val_loader else []
        pseudo_loader = create_pseudo_label_loader(
            self.config, pseudo_csv_path, pseudo_image_dir or image_dir,
            class_names=self.class_names, excluded_images=excluded)
        optimizer, scheduler = self._optimizer(
            self.classification_model, self.config.learning_rate,
            self.config.pseudo_epochs + self.config.num_epochs)
        trainer = Trainer(self.classification_model, str(self.device))
        pseudo_history = trainer.train(pseudo_loader, None, optimizer, nn.CrossEntropyLoss(),
                                       MeanAveragePrecision(), self.config.pseudo_epochs, scheduler=scheduler)
        real_history = trainer.train(train_loader, val_loader, optimizer, nn.CrossEntropyLoss(),
                                     MeanAveragePrecision(), self.config.num_epochs,
                                     checkpoint=self._checkpoint(model_name), scheduler=scheduler)
        history = {'pseudo': pseudo_history, 'real': real_history}
        self.training_history[model_name] = history
        return history

    def train_siamese_model(self, train_csv_path, image_dir, backbone_path, model_name="siamese",
                            val_csv_path=None):
        print(f"Training {model_name} model...")
        self.siamese_model = SiameseNetwork(backbone_path=backbone_path, freeze_backbone=True).to(self.device)
        self.class_names = self.siamese_model.class_names
        if not self.class_names:
            raise ValueError("Backbone checkpoint has no class_names; retrain it with this pipeline")
        self.siamese_model.image_size = self.config.image_size
        train_loader, val_loader = create_pair_loaders(
            self.config, train_csv_path, image_dir, self.class_names, val_csv_path)
        optimizer, scheduler = self._optimizer(self.siamese_model, self.config.pair_model_lr,
                                               self.config.pair_model_epochs)
        history = SiameseTrainer(self.siamese_model, str(self.device)).train(
            train_loader, val_loader, optimizer, nn.BCELoss(), BinaryAccuracy(),
            self.config.pair_model_epochs, checkpoint=self._checkpoint(model_name), scheduler=scheduler)
        self.training_history[model_name] = history
        return history

    def _load_model(self, model_path, model_type):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if checkpoint.get('model_type') != model_type or not checkpoint.get('class_names'):
            raise ValueError("Checkpoint type mismatch or missing label metadata; retrain with this pipeline")
        kwargs = checkpoint['model_kwargs']
        if model_type == 'classification':
            model = WhaleResNet(**kwargs, pretrained=False)
        else:
            model = SiameseNetwork(model_kwargs=kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_names = checkpoint['class_names']
        model.image_size = checkpoint.get('image_size') or self.config.image_size
        return model

    def _gallery(self, model, train_csv_path, image_dir):
        data = read_labels(train_csv_path)
        data = data[data.Id.isin(model.class_names)].reset_index(drop=True)
        dataset = WhaleDataset(image_dir, data, self.config.image_size, transform=False, is_test=True)
        embeddings = []
        for batch in _loader(self.config, dataset):
            embeddings.append(model.backbone(batch['image'].to(self.device))[1])
        mapping = {name: i for i, name in enumerate(model.class_names)}
        labels = torch.tensor([mapping[label] for label in data.Id], device=self.device)
        if set(data.Id) != set(model.class_names):
            raise ValueError("Gallery must contain a reference image for each known whale ID")
        return torch.cat(embeddings), labels

    def _siamese_scores(self, model, images, gallery, labels):
        embeddings = model.backbone(images)[1]
        scores = []
        for query in embeddings:
            values = []
            for chunk in gallery.split(self.config.batch_size):
                values.append(model.score_embeddings(query.expand_as(chunk), chunk).flatten())
            values = torch.cat(values)
            # Max over reference images for each identity; never allocate all image pairs.
            identity_scores = values.new_full((len(model.class_names),), -float('inf'))
            identity_scores.scatter_reduce_(0, labels, values, reduce='amax')
            scores.append(identity_scores)
        return torch.stack(scores)

    def predict(self, test_image_dir, model_path=None, model_type="classification",
                test_csv_path=None, gallery_csv_path=None, gallery_image_dir=None):
        if model_type not in {'classification', 'siamese'}:
            raise ValueError(f"Unknown model type: {model_type}")
        if model_path:
            model = self._load_model(model_path, model_type)
        else:
            model = self.classification_model if model_type == 'classification' else self.siamese_model
        if model is None or not model.class_names:
            raise ValueError("Train or load a model before prediction")
        self.config.image_size = model.image_size
        self.class_names = model.class_names
        model.to(self.device).eval()
        test_loader = create_test_loader(self.config, test_image_dir, test_csv_path)
        predictions, image_names = [], []
        with torch.no_grad():
            if model_type == 'siamese':
                gallery, labels = self._gallery(model, gallery_csv_path or self.config.train_csv,
                                                 gallery_image_dir or self.config.train_images_dir)
            for batch in tqdm(test_loader, desc="Predicting"):
                images = batch['image'].to(self.device)
                if model_type == 'classification':
                    scores = model(images)[0].softmax(dim=1)
                else:
                    scores = self._siamese_scores(model, images, gallery, labels)
                values, indices = scores.topk(min(5, len(model.class_names)), dim=1)
                for row_values, row_indices in zip(values.tolist(), indices.tolist()):
                    ranked = [model.class_names[i] for i in row_indices]
                    threshold = self.config.new_whale_threshold
                    if threshold is not None:
                        position = next((i for i, value in enumerate(row_values) if value < threshold), len(ranked))
                        ranked.insert(position, 'new_whale')
                    predictions.append(' '.join(ranked[:5]))
                image_names.extend(batch['image_name'])
        return pd.DataFrame({'Image': image_names, 'Id': predictions})

    def save_training_history(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as stream:
            json.dump(self.training_history, stream, indent=2)

    def load_training_history(self, filepath):
        with open(filepath) as stream:
            self.training_history = json.load(stream)
