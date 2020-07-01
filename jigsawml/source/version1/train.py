import sys
sys.path.insert(0,'.')
import os
import warnings
warnings.filterwarnings('ignore') 
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, log_loss

DEVICE = 'cuda:0'
    
def reduceLoss(loss):
    shape = loss.shape[0]
    loss = loss.sum() / (shape)
    return loss

def saveModel(model, subset, epoch, loss, metric, path):
    results = {}
    results['model_state_dict'] = model.state_dict()
    results['loss'] = loss
    results['metric'] = metric
    results['epoch'] = epoch
    torch.save(results, path + 'model_{}.pt'.format(subset))
    return None

def validModel(model, data, loss_fn):
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for sample in data:
            label = sample.pop('label').reshape(-1,1)
            label = label.to(DEVICE).squeeze().float()
            for key, value in sample.items(): sample[key] = value.to(DEVICE)
            preds = torch.sigmoid(model(**sample))
            scores.append(preds.cpu().data.numpy())
            labels.append(label.cpu().data.numpy())
            for key, value in sample.items(): sample[key] = value.to('cpu')
        del sample, label, preds
    torch.cuda.empty_cache()
    scores = np.vstack(scores).reshape(-1,1)
    labels = np.vstack(labels).reshape(-1,1)
    loss = round(log_loss(labels, scores),4)
    metric = round(roc_auc_score(labels, scores),4)
    model.train()
    return loss, metric

def trainModel(model, subset, train, valid, loss_fn, optimizer, save, batch, schedular, epochs):
    logfile = save + 'logfile_{}.txt'.format(subset)
    if os.path.exists(logfile): os.remove(logfile)
    logfile = open(logfile, 'w', buffering=1)
    model.train()
    checkpoint = np.NINF
    for epoch in range(epochs):
        step = 0
        optimizer.zero_grad()
        train.dataset.epoch(epoch)
        losses = []
        tq = tqdm(total=len(train) * batch, disable=False)
        for sample in train:
            step += 1
            label = sample.pop('label').squeeze().float()
            label = label.to(DEVICE).reshape(-1,1)
            weight = sample.pop('weight').squeeze().float()
            weight = weight.to(DEVICE).reshape(-1,1)
            for key, value in sample.items(): sample[key] = value.to(DEVICE)
            preds = model(**sample)
            batch_size = preds.shape[0]
            loss = reduceLoss(loss_fn(preds, label, weight))
            loss.backward()
            if step % 4 == 0:
                step = 0
                clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss.cpu().data.item())
            train_loss = round(np.mean(losses),4)
            tq.update(batch_size)
            tq.set_postfix(train_loss='{:.4f}'.format(train_loss))
            for key, value in sample.items(): sample[key] = value.to('cpu')
        del sample, label, preds
        torch.cuda.empty_cache()
        valid_loss, valid_metric = validModel(model, valid, loss_fn)
        schedular.step(valid_loss)
        if checkpoint < valid_metric:
            saveModel(model, subset, epoch, valid_loss, valid_metric, save)
            checkpoint = valid_metric
        tq.set_postfix(val_loss='{:.4f}'.format(valid_loss), val_mtrc ='{:.4f}'.format(valid_metric))
        tq.close()
        logtext  = 'Epoch - {} | '.format(epoch)
        logtext += 'Train Loss - {:.4f} | '.format(train_loss)
        logtext += 'Valid Loss - {:.4f} | '.format(valid_loss)
        logtext += 'Valid Metric - {:.4f} | '.format(valid_metric)
        logtext += '\n'
        logfile.write(logtext)
        sys.stdout.flush()
    logfile.close()
    return None