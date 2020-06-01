import sys
sys.path.insert(0,'..')

import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.metrics import fbeta_score

def makeMask(argsorted, top_n):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask

def binarizePrediction(probabilities, threshold):
    argsorted = probabilities.argsort(axis=1)
    max_mask = makeMask(argsorted, 10)
    min_mask = makeMask(argsorted, 1)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask

def reduceLoss(loss):
    try:
        return loss.sum() / loss.shape[0]
    except:
        return loss

def saveModel(epoch, model, loss, path):
    results = {}
    results['epoch'] = epoch
    results['model_state_dict'] = model.state_dict()
    results['loss'] = loss
    torch.save(results, path)
    return None

def loadModel(model, path):
    results = torch.load(path)
    model.load_state_dict(results['model_state_dict'])
    loss = results['loss']
    print('Model Loaded:', 'Loss:', loss)
    return model, loss

def validModel(model, data, loss_fn, device):
    model.eval()
    losses = []
    label_array = []
    pred_array = []
    for sample in data:
        image = Variable(sample['image'].float().to(device))
        label = Variable(sample['label'].float().squeeze().to(device))
        preds = model(image)
        loss = reduceLoss(loss_fn(preds, label))
        label_array.append(label.cpu().data.numpy())
        pred_array.append(torch.sigmoid(preds).cpu().data.numpy())
        losses.append(loss.data.item())
    del image, label, preds
    torch.cuda.empty_cache()
    label_array = np.rint(np.vstack(label_array)).astype(np.int8)
    pred_array = np.vstack(pred_array)
    metrics = []
    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:
        pred_binary = binarizePrediction(pred_array, 0.15)
        metric = fbeta_score(label_array, pred_binary, beta=2, average='samples')
        metrics += [metric]
    mean_loss = round(np.mean(losses),5)
    metric = round(np.max(metrics),5)
    model.train()
    return mean_loss, metric

def trainModel(model, train_data, valid_data, loss_fn, optimizer, save_path, load_path, epochs, batch, scheduler, device):
    checkpoint = 0.
    if load_path is not None:
        model, checkpoint = loadModel(model, load_path)
    counter = 0
    for epoch in range(epochs):
        model.train()
        tq = tqdm(total=len(train_data) * batch, ncols=0, disable=False)
        losses = []
        label_array = []
        pred_array = []
        step = 0
        alpha = 0.2
        for sample in train_data:
            step += 1
            image = Variable(sample['image'].float().to(device))
            label = Variable(sample['label'].float().squeeze().to(device))
            preds = model(image)
            batch_size = preds.size(0)
            loss = reduceLoss(loss_fn(preds, label))
            (batch_size * loss).backward()
            if step  % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()
                step = 0
            losses.append(loss.data.item())
            train_loss = round(np.mean(losses),5)
            tq.update(image.shape[0])
            tq.set_postfix(trn_ls='{:.5f}'.format(train_loss))
            label_array.append(label.cpu().data.numpy())
            pred_array.append(torch.sigmoid(preds).cpu().data.numpy())
        del image, label, preds
        torch.cuda.empty_cache()
        label_array = np.rint(np.vstack(label_array)).astype(np.int8)
        pred_array = np.vstack(pred_array)
        pred_array = binarizePrediction(pred_array, 0.20)
        train_metric = fbeta_score(label_array, pred_array, beta=2, average='samples')
        valid_loss, valid_metric = validModel(model, valid_data, loss_fn, device)
        postfix = OrderedDict()
        postfix['trn_ls'] = '{:.5f}'.format(train_loss)
        postfix['trn_mt'] = '{:.5f}'.format(train_metric)
        postfix['val_ls'] = '{:.5f}'.format(valid_loss)
        postfix['val_mt'] = '{:.5f}'.format(valid_metric)
        tq.set_postfix(**postfix)
        tq.close()
        scheduler.step()
        if valid_metric > checkpoint:
            counter = 0
            checkpoint = valid_metric
            saveModel(epoch, model, valid_metric, save_path)
        else: counter += 1
        if counter > 5: break
    return None