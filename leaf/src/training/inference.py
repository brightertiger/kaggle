import numpy as np
import torch


def predict(model, data_loader, device=None):
    return predict_with_tta(model, data_loader, device=device, num_tta=1)


def predict_with_tta(model, data_loader, device=None, num_tta=5):
    """Average class probabilities over distinct rotations and mirrored rotations."""
    if not 1 <= num_tta <= 8:
        raise ValueError('num_tta must be between 1 and 8')
    device = device or next(model.parameters()).device
    model.eval()
    predictions = []
    with torch.no_grad():
        for sample in data_loader:
            images = sample[0].float().to(device)
            views = []
            for index in range(num_tta):
                view = torch.rot90(images, index % 4, dims=(-2, -1))
                if index >= 4:
                    view = torch.flip(view, dims=(-1,))
                views.append(model(view).softmax(dim=1))
            predictions.append(torch.stack(views).mean(dim=0).cpu().numpy())
    if not predictions:
        raise ValueError('Prediction loader is empty')
    return np.concatenate(predictions, axis=0)
