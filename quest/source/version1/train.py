import sys
sys.path.insert(0,'.')
import warnings
warnings.filterwarnings('ignore') 
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
from collections import OrderedDict
from scipy.stats import spearmanr
from apex import amp

DEVICE = 'cuda:0'

def reduceLoss(loss):
    shape = loss.shape[0]
    loss = loss.sum() / (shape * 30)
    return loss

def saveModel(model, loss, path):
    results = {}
    results['model_state_dict'] = model.state_dict()
    results['loss'] = loss
    torch.save(results, path)
    return None

def evaluateModel(labels, scores):
    correls = []
    for idx in range(30):
        label = labels.iloc[:,idx]
        score = scores.iloc[:,idx] + np.random.normal(0, 1e-7, scores.shape[0])
        correl = spearmanr(label, score).correlation
        correls.append(correl)
    metric = round(np.nanmean(correls), 4)
    return metric

def validModel(model, data, loss_fn):
    model.eval()
    scores = []
    labels = []
    for sample in data:
        label = sample.pop('label')
        label = label.to(DEVICE).squeeze().float()
        for key, value in sample.items():
            sample[key] = value.to(DEVICE)
        preds = torch.sigmoid(model(**sample))
        scores.append(preds.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())
    del sample, label, preds
    torch.cuda.empty_cache()
    scores = np.vstack(scores)
    labels = np.vstack(labels)
    scores = pd.DataFrame(scores)
    labels = pd.DataFrame(labels)
    mean_loss = -1. * evaluateModel(labels, scores)
    model.train()
    return mean_loss

def trainModel(model, train, valid, loss_fn, optimizer, save, batch, schedular, epochs):
    model.train()
    step = 0
    checkpoint = np.Infinity
    for epoch in range(epochs):
        losses = []
        tq = tqdm(total=len(train) * batch, disable=False)
        for sample in train:
            step += 1
            label = sample.pop('label').squeeze().float()
            label = label.to(DEVICE)
            for key, value in sample.items():
                sample[key] = value.to(DEVICE)
            preds = model(**sample)
            batch_size = preds.shape[0]
            loss = reduceLoss(loss_fn(preds, label))
            with amp.scale_loss(loss, optimizer) as scale_loss:
                 (batch_size * scale_loss).backward()
            if step % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
                step = 0
            losses.append(loss.cpu().data.item())
            train_loss = round(np.mean(losses),4)
            tq.update(batch_size)
            tq.set_postfix(train_loss='{:.4f}'.format(train_loss))
        del sample, label, preds
        torch.cuda.empty_cache()
        valid_loss = validModel(model, valid, loss_fn)
        schedular.step(valid_loss)
        if checkpoint > valid_loss:
            saveModel(model, valid_loss, save)
            checkpoint = valid_loss
        tq.set_postfix(train_loss='{:.4f}'.format(train_loss), valid_loss='{:.4f}'.format(valid_loss))
        tq.close()
    return None