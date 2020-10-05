import sys
sys.path.insert(0,'..')
import os
import torch
import torch.nn as nn
import numpy as np
from apex import amp
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

DEVICE = 'cuda:0'

def reduceMetric(label_array, pred_array):
    label_array = label_array[:,1].reshape(-1,1)
    pred_array = pred_array[:,1].reshape(-1,1)
    metric = round(roc_auc_score(label_array, pred_array),4)
    return metric

def reduceLoss(loss):
    return loss.sum()

def saveModel(epoch, model, loss, path):
    results = {}
    results['epoch'] = epoch
    results['model_state_dict'] = model.state_dict()
    results['loss'] = loss
    torch.save(results, path)
    return None

def validModel(model, data, loss_fn):
    model.eval()
    losses = []
    label_array = []
    pred_array = []
    with torch.no_grad():
        for sample in data:
            image = Variable(sample['image'].float().to(DEVICE))
            label = Variable(sample['label'].float().squeeze().to(DEVICE))
            preds = model(image).squeeze()
            batch_size = label.size(0)
            loss = reduceLoss(loss_fn(preds, label))
            label_array.append(label.cpu().data.numpy())
            pred_array.append(preds.cpu().data.numpy())
            losses.append(loss.data.item())
    label_array = np.vstack(label_array)
    pred_array = np.vstack(pred_array)
    mean_loss = round(np.mean(losses),4)
    mean_metric = round(reduceMetric(label_array, pred_array),4)
    model.train()
    return mean_loss, mean_metric

def trainModel(model, train_data, valid_data, loss_fn, optimizer, save_path, epochs, batch, scheduler):
    checkpoint = 100.
    counter = 0
    logfile = save_path.replace('model_','log_').replace('.pt','.txt')
    if os.path.exists(logfile): os.remove(logfile)
    logfile = open(logfile, 'w', buffering=1)
    for epoch in range(epochs):
        model.train()
        tq = tqdm(total=len(train_data) * batch, ncols=0, disable=False)
        losses = []
        label_array = []
        pred_array = []
        step = 0
        optimizer.zero_grad()
        for sample in train_data:
            step += 1
            image = Variable(sample['image'].float().to(DEVICE))
            label = Variable(sample['label'].float().squeeze().to(DEVICE))
            preds = model(image).squeeze()
            batch_size = label.size(0)
            loss = reduceLoss(loss_fn(preds, label))
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if step % 2 == 0:
                step = 0
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss.data.item())
            train_loss = round(np.mean(losses),4)
            tq.update(batch_size)
            tq.set_postfix(trn_ls='{:.5f}'.format(train_loss))
            label_array.append(label.cpu().data.numpy())
            pred_array.append(preds.cpu().data.numpy())
        label_array = np.vstack(label_array)
        pred_array = np.vstack(pred_array)
        valid_loss , valid_metric = validModel(model, valid_data, loss_fn)
        postfix = OrderedDict()
        postfix['trn_ls'] = '{:.4f}'.format(train_loss)
        postfix['val_ls'] = '{:.4f}'.format(valid_loss)
        postfix['val_mt'] = '{:.4f}'.format(valid_metric)
        tq.set_postfix(**postfix)
        tq.close()
        scheduler.step(valid_loss)
        if valid_loss < checkpoint:
            counter = 0
            checkpoint = valid_loss
            saveModel(epoch, model, valid_metric, save_path)
        else: counter += 1
        logtext  = 'Epoch - {} | '.format(epoch)
        logtext += 'Train Loss - {:.4f} | '.format(train_loss)
        logtext += 'Valid Loss - {:.4f} | '.format(valid_loss)
        logtext += 'Valid Metric - {:.4f} | '.format(valid_metric)
        logtext += '\n'
        logfile.write(logtext)
    logfile.close()
    return None