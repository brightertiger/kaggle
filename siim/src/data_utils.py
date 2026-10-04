from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from .config import Config


def diagnosis_class(row):
    """Preserve auxiliary diagnosis classes; target is authoritative for melanoma."""
    diagnosis = str(row.get('diagnosis', 'other')).lower()
    if 'target' in row and pd.notna(row['target']):
        if int(row['target']) == 1:
            return 1
        if diagnosis == 'melanoma':
            return 0
    if 'keratosis' in diagnosis:
        return 3
    return Config.DIAGNOSIS_LOOKUP.get(diagnosis, 0)


class MelanomaDataset(Dataset):
    def __init__(self, image_path, metadata_df, fold=None, is_training=True, config=None):
        self.config = config or Config()
        self.image_path = Path(image_path)
        self.is_training = is_training
        if fold is not None:
            selected = metadata_df['fold'] != fold if is_training else metadata_df['fold'] == fold
            metadata_df = metadata_df.loc[selected]
        self.data = metadata_df.reset_index(drop=True)
        self.transform = self._get_transforms()
        self.hair_masks = []
        if self.config.HAIR_MASK_DIR is not None:
            mask_dir = Path(self.config.HAIR_MASK_DIR)
            self.hair_masks = sorted(p for p in mask_dir.glob('*')
                                     if p.suffix.lower() in {'.jpg', '.png', '.jpeg'})
            if not self.hair_masks:
                raise FileNotFoundError(f'No hair masks found in {mask_dir}')

    def _get_transforms(self):
        size = self.config.IMAGE_SIZE
        # Resize before cropping so small source images also have valid crop bounds.
        transforms = [A.Resize(height=size, width=size)]
        if self.is_training:
            cutout = min(self.config.CUTOUT_SIZE, size)
            transforms.extend([
                A.RandomSizedCrop(min_max_height=(max(1, round(size * 400 / 512)),
                                                  max(1, round(size * 500 / 512))),
                                  size=(size, size), p=0.5),
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.CoarseDropout(num_holes_range=(1, self.config.CUTOUT_HOLES),
                                hole_height_range=(1, cutout),
                                hole_width_range=(1, cutout), fill=0, p=0.5),
            ])
        transforms.extend([A.Normalize(), ToTensorV2()])
        return A.Compose(transforms, seed=self.config.SEED)

    def _apply_hair_mask(self, image):
        # The migrated code had no mask assets and its all-ones mask blacked out images.
        if self.hair_masks and np.random.random() <= 0.1:
            mask_path = self.hair_masks[np.random.randint(len(self.hair_masks))]
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f'Cannot decode hair mask: {mask_path}')
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            image = cv2.bitwise_and(image, image, mask=(mask > 127).astype(np.uint8))
        return image

    def _encode_metadata(self, row):
        arrays = []
        for column, lookup in [('age_approx', Config.AGE_LOOKUP),
                               ('sex', Config.SEX_LOOKUP),
                               ('anatom_site_general_challenge', Config.ANATOMY_LOOKUP)]:
            encoded = np.zeros(len(lookup), dtype=np.float32)
            index = lookup.get(row.get(column), 0)
            if index:
                encoded[index - 1] = 1
            arrays.append(encoded)
        return np.concatenate(arrays)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row.get('image_name', row.get('image_id'))
        image_path = self.image_path / f'{image_name}.jpg'
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Missing or unreadable image: {image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.is_training:
            image = self._apply_hair_mask(image)
        item = {'image': self.transform(image=image)['image'],
                'metadata': self._encode_metadata(row), 'image_id': image_name}
        if 'diagnosis' in row or 'target' in row:
            label = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
            label[diagnosis_class(row)] = 1.0
            item['label'] = label
        return item


def create_data_loaders(image_path, metadata_df, fold, batch_size=None, num_workers=None, config=None):
    config = config or Config()
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers
    train_dataset = MelanomaDataset(image_path, metadata_df, fold, True, config)
    valid_dataset = MelanomaDataset(image_path, metadata_df, fold, False, config)
    if len(train_dataset) < 2 or len(valid_dataset) == 0 or batch_size < 2:
        raise ValueError('Training needs at least two images, batch_size >= 2, and nonempty validation.')
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)),
                              shuffle=True, num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader


def load_metadata(data_path):
    data_path = Path(data_path)
    frames = []
    for split in ('train', 'test'):
        path = data_path / f'{split}.csv'
        if not path.exists():
            path = data_path / f'{split}_metadata.csv'
        frame = pd.read_csv(path)
        if 'image_name' not in frame and 'image_id' in frame:
            frame = frame.rename(columns={'image_id': 'image_name'})
        required = {'image_name', 'sex', 'age_approx', 'anatom_site_general_challenge'}
        if required - set(frame):
            raise ValueError(f'{path}: missing columns {sorted(required - set(frame))}')
        if frame.empty or frame['image_name'].isna().any() or frame['image_name'].duplicated().any():
            raise ValueError(f'{path}: image_name must be nonempty and unique')
        frame['sex'] = frame['sex'].fillna('unknown')
        # Keep unknown age unknown instead of silently putting it in the youngest bin.
        age = pd.to_numeric(frame['age_approx'], errors='coerce')
        frame['age_approx'] = age.where(age > 0).round(-1).clip(20, 80).fillna(0)
        frame['anatom_site_general_challenge'] = frame['anatom_site_general_challenge'].fillna('unknown')
        if split == 'train':
            if 'target' not in frame:
                if 'diagnosis' not in frame:
                    raise ValueError(f'{path}: training requires target or diagnosis')
                frame['target'] = (frame['diagnosis'] == 'melanoma').astype(int)
            if not frame['target'].isin([0, 1]).all():
                raise ValueError(f'{path}: target must contain binary labels')
        else:
            frame = frame.drop(columns=['target', 'diagnosis'], errors='ignore')
        frames.append(frame)
    return tuple(frames)
