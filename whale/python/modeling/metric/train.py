import os
import torch 
import numpy as np
import albumentations as A
from tqdm import tqdm
from torch import nn 
from torch.autograd import Variable
from prettytable import PrettyTable
from utility.checkpoint import load_model, save_model
from collections import OrderedDict

def logger(name, path, model, rows):
    table = PrettyTable(['Epoch','Train Loss','Train Acc','Valid Loss','Valid Acc'])
    for row in rows: table.add_row(row)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    table_txt = 'Experiment: ' + name + '\n'
    table_txt += 'Total Parameters: ' + str(total_params) + '\n' 
    table_txt += 'Trainable: ' + str(train_params) + '\n'
    table_txt += table.get_string() + '\n'
    if os.path.isfile(path + '/model.log'): os.remove(path + '/model.log')
    with open(path + '/model.log','a+') as file:
        file.write(table_txt)
    file.close()
    return None

def valid_model(model, data, loss_fn, metric_fn, batch):
    model.eval()
    losses = []
    metrics = []
    for sample in data:
        image = Variable(sample['image'].cuda())
        label = Variable(sample['label'].squeeze().cuda())
        preds, _ = model(image)
        loss = loss_fn(preds, label)
        metric = metric_fn(preds, label)
        losses.append(loss.data.item())
        metrics.append(metric.data.item())
    mean_loss = round(np.mean(losses),4)
    mean_metric = round(np.mean(metrics),4)
    model.train()
    return mean_loss, mean_metric

def train_model(name, model, train_data, valid_data, alpha, xent_loss, cent_loss, metric_fn, 
                xent_optim, cent_optim, scheduler, save_path, load_path, epochs, batch):
    if load_path is not None:
        model, checkpoint = load_model(model, load_path + 'model.pth')
    checkpoint = np.Infinity
    log = []
    for epoch in range(epochs):
        model.train()
        tq = tqdm(total=len(train_data) * batch, ncols=0)
        losses = []
        metrics = []
        for sample in train_data:
            if scheduler is not None:
                scheduler.step()
            image = Variable(sample['image'].cuda())
            label = Variable(sample['label'].squeeze().cuda())
            preds, embed = model(image)
            loss = cent_loss(embed, label) * alpha + xent_loss(preds, label)
            xent_optim.zero_grad()
            cent_optim.zero_grad()
            loss.backward()
            for param in cent_loss.parameters():
                param.grad.data *= (1./alpha)
            xent_optim.step()
            cent_optim.step()
            metric = metric_fn(preds, label)
            losses.append(loss.data.item())
            metrics.append(metric.data.item())
            train_loss = round(np.mean(losses),4)
            train_metric = round(np.mean(metrics),4)
            tq.update(image.shape[0])
            tq.set_postfix(trn_ls='{:.2f}'.format(train_loss), trn_ac='{:.2f}'.format(train_metric))
        valid_loss, valid_metric = valid_model(model, valid_data, xent_loss, metric_fn, batch)
        postfix = OrderedDict()
        postfix['trn_ls'] = '{:.2f}'.format(train_loss)
        postfix['trn_ac'] = '{:.2f}'.format(train_metric)
        postfix['val_ls'] = '{:.2f}'.format(valid_loss)
        postfix['val_ac'] = '{:.2f}'.format(valid_metric)
        tq.set_postfix(**postfix)
        tq.close()
        log.append([epoch, train_loss, train_metric, valid_loss, valid_metric])
        if valid_loss < checkpoint:
            checkpoint = valid_loss 
            save_model(epoch, model, valid_loss, save_path + 'model.pth')
        logger(name, save_path, model, log)
    return None
