"""Runnable programmatic example using locally generated sample data."""
from pathlib import Path

import pandas as pd
import torch

from dry_run import make_sample_data, SAMPLE_ROOT, OUTPUT_ROOT
from src.config import Config
from src.data_utils import DiabeticRetinopathyDataset
from src.loss import DiabeticRetinopathyLoss
from src.model import DiabeticRetinopathyModel
from src.optimizer import RAdam
from src.trainer import DiabeticRetinopathyTrainer


def main():
    torch.set_num_threads(2)
    make_sample_data()
    config = Config(DATA_ROOT=str(SAMPLE_ROOT), MODEL_SAVE_PATH=str(OUTPUT_ROOT),
                    MODEL_NAME='efficientnet-b0', PRETRAINED=False, DEVICE='cpu',
                    IMAGE_SIZE=64, NUM_WORKERS=0)
    data = pd.read_csv(Path(config.TRAIN_DATA_PATH) / 'train.csv')
    dataset = DiabeticRetinopathyDataset(Path(config.TRAIN_DATA_PATH) / 'train_images',
                                         data, config.IMAGE_SIZE, config=config)
    model = DiabeticRetinopathyModel(config.MODEL_NAME, config).eval()
    optimizer = RAdam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = DiabeticRetinopathyLoss(mse_weight=config.MSE_WEIGHT, config=config)
    trainer = DiabeticRetinopathyTrainer(config)
    image = dataset[0]['image'].unsqueeze(0)
    with torch.no_grad():
        regression, classification = model(image)
        loss = loss_fn(regression, classification, dataset[0]['label'].reshape(1), torch.ones(1))
    print(f'Image: {tuple(image.shape)}; regression: {tuple(regression.shape)}; '
          f'classification: {tuple(classification.shape)}')
    print(f'Optimizer: {type(optimizer).__name__}; device: {trainer.device}; loss: {loss.item():.4f}')
    print(f'Untrained example severity: {regression.clamp(0, 4).round().item():.0f}')
    print('Run python dry_run.py to train all stages and write a submission.')


if __name__ == '__main__':
    main()
