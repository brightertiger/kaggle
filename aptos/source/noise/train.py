import sys
sys.path.insert(0,'..')
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.metrics import cohen_kappa_score
from apex import amp

DEVICE = 'cuda:0'

def metric_fn(true, pred):
    true = np.rint(true)
    pred = np.rint(pred.clip(0.,4.))
    score = cohen_kappa_score(true, pred, weights='quadratic') 
    return score

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
        image_1 = Variable(sample['image_1'].float().to(DEVICE))
        image_2 = Variable(sample['image_2'].float().to(DEVICE))
        label = Variable(sample['label'].float().squeeze().to(DEVICE))
        weight = Variable(sample['weight'].float().squeeze().to(DEVICE))
        regress_1, classify_1 = model(image_1)
        regress_2, classify_2 = model(image_2)
        batch_size = label.size(0)
        loss = reduceLoss(loss_fn(regress_1, regress_2, classify_1, classify_2, label, weight))
        regress = (regress_1 + regress_2) / 2
        label_array.append(label.cpu().data.numpy())
        pred_array.append(regress.cpu().data.numpy())
        losses.append(loss.data.item())
    del image_1, image_2, label, regress, regress_1, regress_2, classify_1, classify_2
    torch.cuda.empty_cache()
    label_array = np.hstack(label_array)
    pred_array = np.hstack(pred_array)
    metric = metric_fn(label_array, pred_array)
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
            image_1 = Variable(sample['image_1'].float().to(DEVICE))
            image_2 = Variable(sample['image_2'].float().to(DEVICE))
            label = Variable(sample['label'].float().squeeze().to(DEVICE))
            weight = Variable(sample['weight'].float().squeeze().to(DEVICE))
            regress_1, classify_1 = model(image_1)
            regress_2, classify_2 = model(image_2)
            batch_size = label.size(0)
            loss = reduceLoss(loss_fn(regress_1, regress_2, classify_1, classify_2, label, weight))
            with amp.scale_loss(loss, optimizer) as scale_loss:
                (batch_size * scale_loss).backward()
            if step  % 3 == 0:
                optimizer.step()
                optimizer.zero_grad()
                step = 0
            losses.append(loss.data.item())
            train_loss = round(np.mean(losses),5)
            tq.update(batch_size)
            tq.set_postfix(trn_ls='{:.5f}'.format(train_loss))
            regress = (regress_1 + regress_2) / 2
            label_array.append(label.cpu().data.numpy())
            pred_array.append(regress.cpu().data.numpy())
        del image_1, image_2, label, regress, regress_1, regress_2, classify_1, classify_2
        torch.cuda.empty_cache()
        label_array = np.hstack(label_array)
        pred_array = np.hstack(pred_array)
        train_metric = metric_fn(label_array, pred_array)
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