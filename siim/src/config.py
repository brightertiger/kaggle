from pathlib import Path
import torch

class Config:
    SEED = 42
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Data paths
    DATA_DIR = Path('data')
    TRAIN_IMAGES_DIR = DATA_DIR / 'train'
    TEST_IMAGES_DIR = DATA_DIR / 'test'
    METADATA_DIR = DATA_DIR / 'metadata'
    
    # Model paths
    MODEL_DIR = Path('models')
    SCORE_DIR = Path('scores')
    
    # Training parameters
    IMAGE_SIZE = 512
    BATCH_SIZE = 10
    NUM_EPOCHS = 20
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.0
    NUM_WORKERS = 3
    ACCUMULATION_STEPS = 2
    USE_APEX = True  # Used only when NVIDIA Apex and CUDA are available.
    
    # Model architecture
    MODEL_NAME = 'efficientnet-b5'
    NUM_CLASSES = 4
    METADATA_DIM = 13
    PRETRAINED = True
    
    # Data augmentation
    CUTOUT_HOLES = 16
    CUTOUT_SIZE = 64
    HAIR_MASK_DIR = None  # Optional grayscale masks: white keeps pixels, black hides hair.
    
    # Loss function
    POS_WEIGHT = 4.0
    LABEL_SMOOTHING = 0.1
    
    # Cross-validation
    N_FOLDS = 5
    
    # Lookup tables
    SEX_LOOKUP = {'male': 1, 'female': 2}
    AGE_LOOKUP = {20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 70: 6, 80: 7}
    ANATOMY_LOOKUP = {
        'lower extremity': 1, 
        'upper extremity': 2, 
        'torso': 3, 
        'head/neck': 4
    }
    DIAGNOSIS_LOOKUP = {
        'other': 0, 
        'melanoma': 1, 
        'nevus': 2, 
        'keratosis': 3
    }

    def __init__(self, **overrides):
        for name, value in overrides.items():
            if not name.isupper() or not hasattr(type(self), name):
                raise ValueError(f'Unknown configuration option: {name}')
            setattr(self, name, value)

    def as_dict(self):
        # Plain values keep checkpoints compatible with torch.load(weights_only=True).
        return {name: str(value) if isinstance(value, Path) else value
                for name in dir(self) if name.isupper()
                for value in [getattr(self, name)]}
