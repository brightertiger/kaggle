#!/usr/bin/env python3
"""Exercise all supported stages on deterministic, offline, synthetic whale images."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.config import Config
from src.data_utils import create_data_loaders, create_pseudo_label_loader
from src.models import MeanAveragePrecision
from src.pipeline import WhaleIdentificationPipeline


ROOT = Path(__file__).resolve().parent


def make_sample():
    data = ROOT / 'sample_data'
    for folder in ('train', 'test', 'pseudo'):
        (data / folder).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    def save_image(folder, filename, identity, grayscale=False):
        pixels = rng.integers(0, 40, (48, 80, 3), dtype=np.uint8)
        pixels[16:36, 8:72, identity % 3] += 160
        pixels[:, 8 + identity * 9:12 + identity * 9] = 220
        image = Image.fromarray(pixels)
        if grayscale:
            image = image.convert('L')
        image.save(data / folder / filename)

    rows = []
    # Include a singleton identity and new_whale to exercise the split/filter.
    for identity in range(5):
        for example in range(4):
            filename = f'train_{identity}_{example}.jpg'
            save_image('train', filename, identity, grayscale=example == 0)
            rows.append({'Image': filename, 'Id': f'w_{identity:07d}'})
    for identity, label in [(5, 'w_0000005'), (6, 'new_whale')]:
        filename = f'train_{label}.jpg'
        save_image('train', filename, identity)
        rows.append({'Image': filename, 'Id': label})
    pd.DataFrame(rows).to_csv(data / 'train.csv', index=False)
    pseudo_rows = []
    for identity in range(1, 6):
        filename = f'pseudo_{identity}.jpg'
        save_image('pseudo', filename, identity)
        pseudo_rows.append({'Image': filename, 'Id': f'w_{identity:07d}', 'confidence': 0.99})
    pd.DataFrame(pseudo_rows).to_csv(data / 'pseudo_labels.csv', index=False)
    test_names = []
    for identity in range(3):
        filename = f'test_{identity}.jpg'
        save_image('test', filename, identity)
        test_names.append(filename)
    # Deliberately reverse the order to test submission-order preservation.
    pd.DataFrame({'Image': test_names[::-1], 'Id': ['new_whale'] * len(test_names)}).to_csv(
        data / 'sample_submission.csv', index=False)
    return data


def main():
    torch.set_num_threads(2)
    data = make_sample()
    output = ROOT / 'dry_run_output'
    output.mkdir(exist_ok=True)
    config = Config(data_dir=str(data), model_save_dir=str(output / 'models'),
                    device='cpu', num_workers=0, pretrained=False, backbone_name='resnet18',
                    image_size=32, head_dim=32, embedding_dim=16, batch_size=4,
                    num_epochs=1, pseudo_epochs=1, pair_model_epochs=1,
                    weight_decay=0.01, new_whale_threshold=0.5)
    pipeline = WhaleIdentificationPipeline(config)
    train, valid = create_data_loaders(config, config.train_csv, config.train_images_dir)
    assert not set(train.dataset.images) & set(valid.dataset.images)
    assert len(train.dataset.class_names) == 6
    assert 'train_w_0000005.jpg' in train.dataset.images
    # Check the metric against known ranks, independently of model performance.
    metric = MeanAveragePrecision()(torch.tensor([[3., 2., 1.], [3., 2., 1.]]), torch.tensor([0, 1]))
    assert torch.isclose(metric, torch.tensor(0.75))
    pseudo_loader = create_pseudo_label_loader(
        config, str(data / 'pseudo_labels.csv'), str(data / 'pseudo'),
        class_names=train.dataset.class_names)
    assert pseudo_loader.dataset.labels == [1, 2, 3, 4, 5]
    pipeline.train_classification_model(config.train_csv, config.train_images_dir)
    pipeline.train_with_pseudo_labels(config.train_csv, str(data / 'pseudo_labels.csv'),
                                      config.train_images_dir, pseudo_image_dir=str(data / 'pseudo'))
    pipeline.train_classification_model(config.train_csv, config.train_images_dir,
                                        use_center_loss=True, model_name='center_loss')
    center_path = output / 'models' / 'center_loss' / 'model.pth'
    center_checkpoint = torch.load(center_path, map_location='cpu', weights_only=True)
    assert 'center_loss_state_dict' in center_checkpoint
    center_param_id = center_checkpoint['optimizer_state_dict']['param_groups'][0]['params'][-1]
    assert center_checkpoint['optimizer_state_dict']['state'][center_param_id]['step'] > 0
    assert pipeline.center_loss.centers.grad is not None
    assert torch.isfinite(pipeline.center_loss.centers.grad).all()
    pipeline.train_siamese_model(config.train_csv, config.train_images_dir, str(center_path))
    for name, value in pipeline.siamese_model.backbone.state_dict().items():
        assert torch.equal(value.cpu(), center_checkpoint['model_state_dict'][name]), name
    test_csv = str(data / 'sample_submission.csv')
    expected_names = pd.read_csv(test_csv).Image.tolist()
    allowed = set(pipeline.class_names) | {'new_whale'}
    for mode in ('classification', 'pseudo_label', 'center_loss', 'siamese'):
        # A fresh pipeline must recover architecture and vocabulary from disk.
        fresh = WhaleIdentificationPipeline(config)
        result = fresh.predict(config.test_images_dir,
                               model_path=str(output / 'models' / mode / 'model.pth'),
                               model_type='siamese' if mode == 'siamese' else 'classification',
                               test_csv_path=test_csv)
        assert result.Image.tolist() == expected_names
        assert list(result.columns) == ['Image', 'Id']
        for value in result.Id:
            labels = value.split()
            assert len(labels) == len(set(labels)) == 5
            assert set(labels) <= allowed
        filename = 'submission.csv' if mode == 'center_loss' else f'{mode}_submission.csv'
        result.to_csv(output / filename, index=False)
    pipeline.save_training_history(str(output / 'training_history.json'))
    history = json.loads((output / 'training_history.json').read_text())
    assert len(history['pseudo_label']['pseudo']['train_losses']) == 1
    assert len(history['pseudo_label']['real']['train_losses']) == 1
    for stage in history.values():
        for run in (stage.values() if 'pseudo' in stage else [stage]):
            assert all(np.isfinite(values).all() for values in run.values())
    print('\nDry run passed: synthetic Image/Id CSVs and RGB/grayscale JPEGs; CPU preprocessing,')
    print('classification, pseudo-label fine-tuning, center loss, Siamese pairs/gallery scoring,')
    print('checkpoint reloads and submission writing. No stages skipped; no weights downloaded.')
    print(f'Output: {output / "submission.csv"}')


if __name__ == '__main__':
    main()
