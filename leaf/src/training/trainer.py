"""Training with accumulation, validation checkpoints and optional SWA."""
from pathlib import Path
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from ..utils.config import Config


def reduce_metric(label_array, pred_array):
    return float((label_array.reshape(-1) == pred_array.argmax(axis=1)).mean())


def reduce_loss(loss):
    return loss.sum()


def save_model(epoch, model, loss, path, metadata=None, accuracy=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'loss': float(loss), 'accuracy': accuracy, 'metadata': metadata or {}}, path)


def validate_model(model, data_loader, loss_fn, device=None):
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_loss, total = 0.0, 0
    labels, predictions = [], []
    with torch.no_grad():
        for images, targets in data_loader:
            targets = targets.long().reshape(-1).to(device)
            logits = model(images.float().to(device))
            total_loss += loss_fn(logits, targets).item() * len(targets)
            total += len(targets)
            labels.append(targets.cpu().numpy())
            predictions.append(logits.cpu().numpy())
    model.train(was_training)
    if not total:
        raise ValueError('Validation loader is empty')
    return total_loss / total, reduce_metric(np.concatenate(labels), np.concatenate(predictions))


def train_model(model, train_loader, valid_loader, loss_fn, optimizer, save_path,
                epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, scheduler=None,
                device=None, accumulation_steps=4, swa_start=7, patience=None, metadata=None):
    # batch_size remains accepted for callers; actual batch sizes drive averaging.
    device = device or next(model.parameters()).device
    if epochs < 1 or accumulation_steps < 1 or not len(train_loader):
        raise ValueError('Training requires positive epochs, accumulation steps and a nonempty loader')
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    swa_path = save_path.with_name(save_path.stem + '_swa.pt')
    # A rerun must not accidentally select an SWA checkpoint from an older run.
    swa_path.unlink(missing_ok=True)
    best_metric, counter = -float('inf'), 0
    swa_model = AveragedModel(model) if swa_start < epochs else None
    swa_scheduler = SWALR(optimizer, swa_lr=1e-6) if swa_model is not None else None
    history = []
    with save_path.with_suffix('.log').open('w', buffering=1) as logfile:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total, sample_total, group_samples = 0.0, 0, 0
            for step, (images, targets) in enumerate(train_loader, start=1):
                targets = targets.long().reshape(-1).to(device)
                logits = model(images.float().to(device))
                loss = loss_fn(logits, targets)
                # Normalize by actual examples, including an incomplete final group.
                (loss * len(targets)).backward()
                group_samples += len(targets)
                if step % accumulation_steps == 0 or step == len(train_loader):
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.div_(group_samples)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    group_samples = 0
                loss_total += loss.item() * len(targets)
                sample_total += len(targets)
            valid_loss, valid_metric = validate_model(model, valid_loader, loss_fn, device)
            if valid_metric > best_metric:
                counter, best_metric = 0, valid_metric
                save_model(epoch, model, valid_loss, save_path, metadata, valid_metric)
            else:
                counter += 1
            if swa_model is not None and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            elif scheduler is not None:
                scheduler.step()
            record = dict(epoch=epoch, train_loss=loss_total / sample_total,
                          valid_loss=valid_loss, accuracy=valid_metric)
            history.append(record)
            line = f'Epoch {epoch}: train_loss={record["train_loss"]:.4f}, valid_loss={valid_loss:.4f}, accuracy={valid_metric:.4f}'
            print(line)
            logfile.write(line + '\n')
            if patience is not None and counter >= patience:
                break
    if swa_model is not None and int(swa_model.n_averaged) > 0:
        update_bn(train_loader, swa_model, device=device)
        swa_loss, swa_metric = validate_model(swa_model, valid_loader, loss_fn, device)
        # Save the underlying model so standard and SWA files load identically.
        save_model(epoch, swa_model.module, swa_loss, swa_path, metadata, swa_metric)
    return history
