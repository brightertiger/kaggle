class Config:
    SEED = 42
    NUM_CLASSES = 5
    IMAGE_SIZE = 512
    BATCH_SIZE = 6
    NUM_WORKERS = 2
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    DEVICE = 'cuda:0'
    
    MODEL_NAME = 'tf_efficientnet_b4_ns'
    
    TRAIN_TRANSFORMS = [
        'PadIfNeeded',
        'Resize', 
        'RandomResizedCrop',
        'Transpose',
        'HorizontalFlip',
        'VerticalFlip',
        'RandomRotate90',
        'ShiftScaleRotate',
        'RandomBrightnessContrast',
        'HueSaturationValue',
        'Normalize',
        'CoarseDropout',
        'Cutout'
    ]
    
    VALID_TRANSFORMS = [
        'PadIfNeeded',
        'Resize',
        'CenterCrop',
        'Normalize'
    ]
    
    CLASS_NAMES = [
        'Cassava Bacterial Blight (CBB)',
        'Cassava Brown Streak Disease (CBSD)', 
        'Cassava Green Mottle (CGM)',
        'Cassava Mosaic Disease (CMD)',
        'Healthy'
    ]
