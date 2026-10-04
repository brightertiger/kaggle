from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ..utils.config import Config


def get_transforms(is_train, image_size=512, seed=42):
    # Scale the original 600 x 800 padding/resizing canvas with the crop size.
    height = max(image_size, round(image_size * 600 / 512))
    width = max(image_size, round(image_size * 800 / 512))
    transforms = [
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_REFLECT_101),
        A.Resize(height=height, width=width),
    ]
    if is_train:
        transforms += [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
            A.Transpose(p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.5),
            # CoarseDropout replaces the removed Cutout transform.
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.05, 0.15),
                            hole_width_range=(0.05, 0.15), p=0.5),
        ]
    else:
        transforms += [A.CenterCrop(height=image_size, width=image_size)]
    transforms += [A.Normalize(), ToTensorV2()]
    return A.Compose(transforms, seed=seed)


class CassavaDataset(Dataset):
    def __init__(self, image_path, data, is_train=True, image_size=512, seed=42):
        self.image_path = Path(image_path)
        self.data = (pd.read_csv(data) if isinstance(data, (str, Path)) else data).reset_index(drop=True)
        self.transforms = get_transforms(is_train, image_size, seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        path = self.image_path / row['image_id']
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f'Cannot read image: {path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        # Inference has no labels; the placeholder keeps the same loader interface.
        return image, int(row['label']) if 'label' in row else -1


def create_data_loaders(image_path, label_path, fold, config=None):
    cfg = config or Config()
    data = pd.read_csv(label_path)
    train = data[data['fold'] != fold]
    valid = data[data['fold'] == fold]
    if train.empty or valid.empty:
        raise ValueError(f'Fold {fold} has an empty training or validation split')
    kwargs = dict(batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                  pin_memory=str(cfg.DEVICE).startswith('cuda'))
    train_set = CassavaDataset(image_path, train, True, cfg.IMAGE_SIZE, cfg.SEED)
    valid_set = CassavaDataset(image_path, valid, False, cfg.IMAGE_SIZE, cfg.SEED)
    return (
        DataLoader(train_set, shuffle=True, generator=torch.Generator().manual_seed(cfg.SEED), **kwargs),
        DataLoader(valid_set, shuffle=False, **kwargs),
    )
