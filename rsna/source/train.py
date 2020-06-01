import sys
sys.path.insert(0,'..')
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F

from apex import amp

DEVICE = 'cuda:0'

def metric_fn(preds, true):
    preds = torch.from_numpy(preds)
    true = torch.from_numpy(true)
    weight = torch.ones_like(preds)
    weight[:,0] += 1 
    metric = F.binary_cross_entropy_with_logits(preds, true, weight=weight, reduction='mean')
    metric = metric.data.item()
    return metric
    
def reduceLoss(loss):
    return loss.sum() / loss.shape[0]

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
    for sample in data:
        image = Variable(sample['image'].float().to(DEVICE))
        label = Variable(sample['label'].float().squeeze().to(DEVICE))
        preds = model(image)
        batch_size = label.size(0)
        loss = reduceLoss(loss_fn(preds, label))
        label_array.append(label.cpu().data.numpy())
        pred_array.append(preds.cpu().data.numpy())
        losses.append(loss.data.item())
    del image, label, preds
    torch.cuda.empty_cache()
    label_array = np.hstack(label_array)
    pred_array = np.hstack(pred_array)
    metric = metric_fn(pred_array, label_array)
    mean_loss = round(np.mean(losses),5)
    metric = round(metric,5)
    model.train()
    return mean_loss, metric

def trainModel(model, train_data, valid_data, loss_fn, optimizer, save_path, epochs, batch, scheduler):
    checkpoint = 100.
    counter = 0
    for epoch in range(epochs):
        model.train()
        tq = tqdm(total=len(train_data) * batch, ncols=0, disable=False)
        losses = []
        label_array = []
        pred_array = []
        step = 0
        for sample in train_data:
            step += 1
            image = Variable(sample['image'].float().to(DEVICE))
            label = Variable(sample['label'].float().squeeze().to(DEVICE))
            preds = model(image)
            batch_size = label.size(0)
            loss = reduceLoss(loss_fn(preds, label))
            with amp.scale_loss(loss, optimizer) as scale_loss:
                (batch_size * scale_loss).backward()
            if step  % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()
                step = 0
            losses.append(loss.data.item())
            train_loss = round(np.mean(losses),5)
            tq.update(batch_size)
            tq.set_postfix(trn_ls='{:.5f}'.format(train_loss))
            label_array.append(label.cpu().data.numpy())
            pred_array.append(preds.cpu().data.numpy())
        del image, label, preds
        torch.cuda.empty_cache()
        label_array = np.hstack(label_array)
        pred_array = np.hstack(pred_array)
        train_metric = metric_fn(pred_array, label_array)
        valid_loss, valid_metric = validModel(model, valid_data, loss_fn)
        postfix = OrderedDict()
        postfix['trn_ls'] = '{:.5f}'.format(train_loss)
        postfix['trn_mt'] = '{:.5f}'.format(train_metric)
        postfix['val_ls'] = '{:.5f}'.format(valid_loss)
        postfix['val_mt'] = '{:.5f}'.format(valid_metric)
        tq.set_postfix(**postfix)
        tq.close()
        scheduler.step()
        if valid_loss < checkpoint:
            counter = 0
            checkpoint = valid_loss
            saveModel(epoch, model, valid_loss, save_path)
        else: counter += 1
    return None