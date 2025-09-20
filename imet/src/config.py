import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Config:
    data_path: str = '../data'
    output_path: str = '../output'
    model_name: str = 'resnext101'
    batch_size: int = 20
    epochs: int = 20
    learning_rate: float = 1e-4
    image_size: int = 300
    num_folds: int = 10
    fold_idx: Optional[int] = None
    freeze_backbone: bool = False
    device: str = 'auto'
    num_workers: int = 6
    patience: int = 5
    seed: int = 2017
    
    num_classes: int = 1103
    epsilon: float = 0.1
    focal_gamma: float = 1.0
    f2_beta: float = 2.0
    
    train_transforms: Dict[str, Any] = field(default_factory=lambda: {
        'random_horizontal_flip': 0.5,
        'random_crop': True,
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    })
    
    val_transforms: Dict[str, Any] = field(default_factory=lambda: {
        'random_crop': True,
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    })
    
    thresholds: List[float] = field(default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30])
    top_k: int = 10
    min_threshold: float = 0.2
    
    def __post_init__(self):
        self._setup_paths()
        self._setup_device()
        self._setup_directories()
    
    def _setup_paths(self):
        self.train_csv_path = os.path.join(self.data_path, 'train.csv.zip')
        self.folds_csv_path = os.path.join(self.data_path, 'folds.csv')
        self.subset_csv_path = os.path.join(self.data_path, 'subset.csv')
        self.train_images_path = os.path.join(self.data_path, 'train')
        self.test_images_path = os.path.join(self.data_path, 'test')
        self.sample_submission_path = os.path.join(self.data_path, 'sample_submission.csv')
        
        self.model_dir = os.path.join(self.output_path, 'models')
        self.score_dir = os.path.join(self.output_path, 'scores')
        self.submission_dir = os.path.join(self.output_path, 'submissions')
        self.logs_dir = os.path.join(self.output_path, 'logs')
    
    def _setup_device(self):
        if self.device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            self.device = 'cpu'
    
    def _setup_directories(self):
        for directory in [self.model_dir, self.score_dir, 
                         self.submission_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
    
    def get_model_path(self, fold: int, stage: str = 'stage_2') -> str:
        return os.path.join(self.model_dir, f'{self.model_name}_{stage}_{fold}.pt')
    
    def get_score_path(self, fold: int) -> str:
        return os.path.join(self.score_dir, f'score_{fold}.csv.gz')
    
    def get_submission_path(self, name: str = 'submission') -> str:
        return os.path.join(self.submission_dir, f'{name}.csv')
    
    def get_log_path(self, fold: int) -> str:
        return os.path.join(self.logs_dir, f'training_fold_{fold}.log')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'data_path': self.data_path,
            'output_path': self.output_path,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'image_size': self.image_size,
            'num_folds': self.num_folds,
            'fold_idx': self.fold_idx,
            'freeze_backbone': self.freeze_backbone,
            'device': self.device,
            'num_workers': self.num_workers,
            'patience': self.patience,
            'seed': self.seed,
            'num_classes': self.num_classes,
            'epsilon': self.epsilon,
            'focal_gamma': self.focal_gamma,
            'f2_beta': self.f2_beta
        }
    
    def __str__(self) -> str:
        config_str = "Configuration:\n"
        for key, value in self.to_dict().items():
            config_str += f"  {key}: {value}\n"
        return config_str
