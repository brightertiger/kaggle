import os
import torch 
import pandas as pd
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
        image_1 = Variable(sample['image_1'].cuda())
        image_2 = Variable(sample['image_2'].cuda())
        label = Variable(sample['label'].float().squeeze().cuda())
        preds = model(image_1, image_2).squeeze()
        loss = loss_fn(preds, label)
        metric = metric_fn(preds, label)
        losses.append(loss.data.item())
        metrics.append(metric.data.item())
    mean_loss = round(np.mean(losses),4)
    mean_metric = round(np.mean(metrics),4)
    model.train()
    return mean_loss, mean_metric

def train_model(name, model, train_data, valid_data, loss_fn, metric_fn, 
                optimizer, scheduler, save_path, load_path, epochs, batch):
    if load_path is not None:
        model, checkpoint = load_model(model, load_path + 'model.pth')
    checkpoint = np.Infinity
    log = []
    for epoch in range(epochs):
        model.train()
        tq = tqdm(total=len(train_data) * batch, ncols=0, disable=False)
        losses = []
        metrics = []
        for sample in train_data:
            if scheduler is not None:
                scheduler.step()
            image_1 = Variable(sample['image_1'].cuda())
            image_2 = Variable(sample['image_2'].cuda())
            label = Variable(sample['label'].float().squeeze().cuda())
            preds = model(image_1, image_2).squeeze()
            loss = loss_fn(preds, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metric = metric_fn(preds, label)
            losses.append(loss.data.item())
            metrics.append(metric.data.item())
            train_loss = round(np.mean(losses),4)
            train_metric = round(np.mean(metrics),4)
            tq.update(image_1.shape[0])
            tq.set_postfix(trn_ls='{:.2f}'.format(train_loss), trn_ac='{:.2f}'.format(train_metric))
        valid_loss, valid_metric = valid_model(model, valid_data, loss_fn, metric_fn, batch)
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

def score_model(model, data, batch, path):
    model, checkpoint = load_model(model, path + 'model.pth')
    model.eval()
    scores = pd.DataFrame([])
    tq = tqdm(total=len(data) * batch, ncols=0)
    preds_1 = []
    preds_2 = []
    index = []
    labels = []
    for sample in data:
        image_1 = Variable(sample['image_1'].float().cuda())
        image_2 = Variable(sample['image_2'].float().cuda())
        name = np.array(sample['idx']).reshape(-1,1)
        label = np.array(sample['label']).reshape(-1,1)
        pred_1 = model(image_1, image_2).cpu().data.numpy()
        pred_2 = model(image_2, image_1).cpu().data.numpy()
        preds_1.append(pred_1)
        preds_2.append(pred_2)
        labels.append(label)
        index.append(name)
        tq.update(pred_1.shape[0])
    tq.close()
    preds_1 = np.vstack(preds_1)
    preds_2 = np.vstack(preds_2)
    labels = np.vstack(labels)
    index = np.vstack(index)
    index = pd.DataFrame(index, columns=['Image'])
    labels = pd.DataFrame(labels, columns=['Id'])
    preds_1 = pd.DataFrame(preds_1, columns=['Score_1'])
    preds_2 = pd.DataFrame(preds_2, columns=['Score_2'])
    scores = index.join(labels).join(preds_1).join(preds_2)
    print('Records:', scores.shape[0])
    scores.to_csv(path + 'scores.csv', index=False)
    return None