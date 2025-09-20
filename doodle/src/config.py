import os
import torch

class Config:
    def __init__(self):
        self.data_path = '../data'
        self.model_path = os.path.join(self.data_path, 'model')
        self.score_path = os.path.join(self.data_path, 'score')
        self.submit_path = os.path.join(self.data_path, 'submit')
        
        self.image_size = 64
        self.batch_size = 650
        self.num_workers = 11
        self.num_classes = 340
        
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.epochs = 50
        self.patience = 5
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = 2017
        
        self.models = {
            'resnet18': {'name': 'ResNet18', 'fc_features': 512},
            'resnet34': {'name': 'ResNet34', 'fc_features': 512},
            'resnet50': {'name': 'ResNet50', 'fc_features': 2048}
        }
