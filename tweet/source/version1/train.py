import sys
sys.path.insert(0,'.')
import os
import warnings
warnings.filterwarnings('ignore')
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from collections import OrderedDict

DEVICE = 'cuda:0'
    
def reduceLoss(loss):
    # shape = loss.shape[0]
    # loss = loss.sum() / (shape)
    return loss

def saveModel(model, subset, epoch, loss, path):
    results = {}
    results['model_state_dict'] = model.state_dict()
    results['loss'] = loss
    results['epoch'] = epoch
    torch.save(results, path + 'model_{}.pt'.format(subset))
    return None

def validModel(model, data, loss_fn):
    model.eval()
    losses = []
    for sample in data:
        for key, value in sample.items(): sample[key] = value.to(DEVICE)
        s_pred, e_pred, pred = model(sample['tokens'], sample['masks'])
        s_tok, e_tok, label = sample['start_idx'], sample['end_idx'], sample['aux_label']
        loss = reduceLoss(loss_fn(s_pred, e_pred, s_tok, e_tok, pred, label))
        losses.append(loss.cpu().data.item())
        for key, value in sample.items(): sample[key] = value.to('cpu')
    del sample
    torch.cuda.empty_cache()
    loss = round(np.mean(losses),4)
    model.train()
    return loss

def trainModel(model, subset, train, valid, loss_fn, optimizer, save, batch, schedular, epochs):
    logfile = save + 'logfile_{}.txt'.format(subset)
    if os.path.exists(logfile): os.remove(logfile)
    logfile = open(logfile, 'w', buffering=1)
    model.train()
    checkpoint = np.PINF
    for epoch in range(epochs):
        step = 0
        optimizer.zero_grad()
        losses = []
        tq = tqdm(total=len(train) * batch, disable=False)
        for sample in train:
            step += 1
            for key, value in sample.items(): sample[key] = value.to(DEVICE)
            batch_size = sample['tokens'].shape[0]
            s_pred, e_pred, pred = model(sample['tokens'], sample['masks'])
            s_tok, e_tok, label = sample['start_idx'], sample['end_idx'], sample['aux_label']
            loss = reduceLoss(loss_fn(s_pred, e_pred, s_tok, e_tok, pred, label))
            loss.backward()
            if step % 8 == 0:
                step = 0
                clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss.cpu().data.item())
            train_loss = round(np.mean(losses),4)
            tq.update(batch_size)
            tq.set_postfix(train_loss='{:.4f}'.format(train_loss))
            for key, value in sample.items(): sample[key] = value.to('cpu')
        del sample
        torch.cuda.empty_cache()
        valid_loss = validModel(model, valid, loss_fn)
        schedular.step(valid_loss)
        if checkpoint > valid_loss:
            saveModel(model, subset, epoch, valid_loss, save)
            checkpoint = valid_loss
        tq.set_postfix(trn_loss='{:.4f}'.format(train_loss), val_loss ='{:.4f}'.format(valid_loss))
        tq.close()
        logtext  = 'Epoch - {} | '.format(epoch)
        logtext += 'Train Loss - {:.4f} | '.format(train_loss)
        logtext += 'Valid Loss - {:.4f} | '.format(valid_loss)
        logtext += '\n'
        logfile.write(logtext)
        sys.stdout.flush()
    logfile.close()
    return None