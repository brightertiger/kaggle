import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from ..utils.config import Config

def reduce_metric(label_array, pred_array):
    label_array = label_array.reshape(-1, 1)
    pred_array = pred_array.argmax(axis=1).reshape(-1, 1)
    metric = round((label_array == pred_array).mean(), 4)
    return metric

def reduce_loss(loss):
    return loss.sum()

def save_model(epoch, model, loss, path):
    results = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }
    torch.save(results, path)

def validate_model(model, data_loader, loss_fn, device=Config.DEVICE):
    model.eval()
    losses = []
    label_array = []
    pred_array = []
    
    with torch.no_grad():
        for sample in data_loader:
            image = Variable(sample[0].float().to(device))
            label = Variable(sample[1].long().squeeze().to(device))
            
            preds = model(image).squeeze()
            batch_size = label.size(0)
            loss = reduce_loss(loss_fn(preds, label))
            
            label_array.append(label.cpu().data.numpy())
            pred_array.append(preds.cpu().data.numpy())
            losses.append(loss.data.item())
    
    label_array = np.hstack(label_array)
    pred_array = np.vstack(pred_array)
    mean_loss = round(np.mean(losses), 4)
    mean_metric = round(reduce_metric(label_array, pred_array), 4)
    
    model.train()
    return mean_loss, mean_metric

def train_model(model, train_loader, valid_loader, loss_fn, optimizer, save_path, 
                epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, scheduler=None):
    checkpoint = -100.
    counter = 0
    
    logfile = save_path.replace('model_', 'log_').replace('.pt', '.txt')
    if os.path.exists(logfile):
        os.remove(logfile)
    logfile = open(logfile, 'w', buffering=1)
    
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-6)
    
    for epoch in range(epochs):
        model.train()
        tq = tqdm(total=len(train_loader) * batch_size, ncols=0, disable=False)
        losses = []
        label_array = []
        pred_array = []
        step = 0
        
        optimizer.zero_grad()
        
        for sample in train_loader:
            step += 1
            image = Variable(sample[0].float().to(Config.DEVICE))
            label = Variable(sample[1].long().squeeze().to(Config.DEVICE))
            
            batch_size = label.size(0)
            preds = model(image).squeeze()
            loss = loss_fn(preds, label)
            loss.backward()
            
            if step % 4 == 0:
                optimizer.step()
                model.zero_grad()
            
            losses.append(loss.data.item())
            train_loss = round(np.mean(losses), 4)
            tq.update(batch_size)
            tq.set_postfix(trn_ls='{:.5f}'.format(train_loss))
            
            label_array.append(label.cpu().data.numpy())
            pred_array.append(preds.cpu().data.numpy())
        
        label_array = np.vstack(label_array)
        pred_array = np.vstack(pred_array)
        
        valid_loss, valid_metric = validate_model(model, valid_loader, loss_fn)
        
        postfix = OrderedDict()
        postfix['trn_ls'] = '{:.4f}'.format(train_loss)
        postfix['val_ls'] = '{:.4f}'.format(valid_loss)
        postfix['val_mt'] = '{:.4f}'.format(valid_metric)
        tq.set_postfix(**postfix)
        tq.close()
        
        if valid_metric > checkpoint:
            counter = 0
            checkpoint = valid_metric
            save_model(epoch, model, valid_metric, save_path)
        else:
            counter += 1
        
        logtext = f'Epoch - {epoch} | '
        logtext += f'Train Loss - {train_loss:.4f} | '
        logtext += f'Valid Loss - {valid_loss:.4f} | '
        logtext += f'Valid Metric - {valid_metric:.4f} | \n'
        logfile.write(logtext)
        
        if epoch >= 7:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            if scheduler:
                scheduler.step()
    
    update_bn(train_loader, swa_model, Config.DEVICE)
    save_model(epoch, swa_model, valid_metric, save_path.replace('.pt', '_swa.pt'))
    logfile.close()
