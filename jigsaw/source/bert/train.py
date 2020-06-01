import sys
sys.path.insert(0,'.')
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
from collections import OrderedDict
from .evaluate import *

valid = pd.read_csv('/workspace/data/train.csv')

def evalauteModel(score, save):
    score.to_csv(save + 'score.csv', index=False)
    score = score.merge(valid, on='id')
    bias_metrics = compute_bias_metrics_for_model(score)
    bias_metrics.to_csv(save + 'metrics.csv', index=False)
    metric = get_final_metric(bias_metrics, calculate_overall_auc(score))
    return round(metric, 5)

def saveModel(model, loss, path):
    results = {}
    results['model_state_dict'] = model.state_dict()
    results['loss'] = loss
    torch.save(results, path)
    return None

def validModel(device, model, data, save):
    model.eval()
    score = []
    idx = []
    for sample in data:
        idx += sample['idx']
        text = Variable(sample['text'].to(device))
        preds, _ = model(text)
        preds = torch.sigmoid(preds).cpu().data.numpy()
        score.append(preds)
    del text, preds
    torch.cuda.empty_cache()
    score = np.vstack(score)
    idx = [y.item() for x in idx for y in x]
    score = pd.DataFrame(score, columns=['prediction'])
    score['id'] = idx
    score = score[['id','prediction']]
    mean_loss = evalauteModel(score, save)
    model.train()
    return mean_loss

def trainModel(device, model, train, valid, loss_fn, optimizer, save, batch):
    model.train()
    tq = tqdm(total=len(train) * batch, disable=False)
    losses = []
    step = 0
    valid_loss = -0.01
    counter = 0
    for sample in train:
        counter += 12
        step += 1
        text = Variable(sample['text'].squeeze().long().to(device))
        weight = Variable(sample['weight'].float().squeeze().to(device))
        label = Variable(sample['labels'].float().squeeze().to(device))
        aux_label = Variable(sample['aux'].float().squeeze().to(device))
        preds, aux = model(text)
        batch_size = preds.shape[0]
        loss = loss_fn(preds, label, aux, aux_label, weight)
        loss.backward()
        if step % 5 == 0:
            optimizer.step()
            optimizer.zero_grad()
            step = 0
        losses.append(loss.data.item())
        train_loss = round(np.mean(losses),4)
        tq.update(batch_size)
        tq.set_postfix(train_loss='{:.4f}'.format(train_loss), valid_loss='{:.4f}'.format(valid_loss))
    del text, weight, label, preds
    torch.cuda.empty_cache() 
    saveModel(model, train_loss, save + 'model.pt')
    valid_loss = validModel(device, model, valid, save)
    saveModel(model, valid_loss, save + 'model.pt')
    tq.set_postfix(train_loss='{:.4f}'.format(train_loss), valid_loss='{:.4f}'.format(valid_loss))
    tq.close()
    return None
