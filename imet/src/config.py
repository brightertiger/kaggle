import os
import torch
from dataclasses import asdict, dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Config:
    data_path: str = './data'
    output_path: str = './output'
    model_name: str = 'resnext101'
    batch_size: int = 20
    epochs: int = 20
    learning_rate: float = 1e-4
    image_size: int = 300
    num_folds: int = 10
    fold_idx: Optional[int] = None
    freeze_backbone: bool = False
    pretrained: bool = True
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
        if self.num_folds < 2:
            raise ValueError('num_folds must be at least 2')
        if self.fold_idx is not None and not 1 <= self.fold_idx <= self.num_folds:
            raise ValueError('fold_idx must be between 1 and num_folds')
        if min(self.batch_size, self.epochs, self.image_size, self.num_classes, self.top_k, self.patience) < 1:
            raise ValueError('Batch size, epochs, image size, classes, top_k and patience must be positive')
        if not 0 <= self.epsilon < 0.5:
            raise ValueError('epsilon must be in [0, 0.5)')
        if not self.thresholds or self.learning_rate <= 0 or self.num_workers < 0:
            raise ValueError('Invalid thresholds, learning rate or worker count')
        self._setup_paths()
        self._setup_device()
        self._setup_directories()
    
    def _setup_paths(self):
        plain_csv = os.path.join(self.data_path, 'train.csv')
        self.train_csv_path = plain_csv if os.path.exists(plain_csv) else os.path.join(self.data_path, 'train.csv.zip')
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
        aliases = {'model': 'model_name', 'lr': 'learning_rate', 'folds': 'num_folds'}
        for key, value in vars(args).items():
            key = aliases.get(key, key)
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        self.__post_init__()
    
    def get_model_path(self, fold: int, stage: str = 'stage_2') -> str:
        return os.path.join(self.model_dir, f'{self.model_name}_{stage}_{fold}.pt')
    
    def get_score_path(self, fold: int) -> str:
        return os.path.join(self.score_dir, f'score_{fold}.csv.gz')
    
    def get_submission_path(self, name: str = 'submission') -> str:
        return os.path.join(self.submission_dir, f'{name}.csv')
    
    def get_log_path(self, fold: int) -> str:
        return os.path.join(self.logs_dir, f'training_fold_{fold}.log')
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def __str__(self) -> str:
        config_str = "Configuration:\n"
        for key, value in self.to_dict().items():
            config_str += f"  {key}: {value}\n"
        return config_str
