import torch 
import torch.nn as nn
import pandas as pd
import numpy as np

def scoreModel(model, data, save):
    model.eval()
    scores = []
    labels = []
    for sample in data:
        label = sample.pop('label')
        label = label.cuda().squeeze().float()
        for key, value in sample.items():
            sample[key] = value.cuda() 
        preds = torch.sigmoid(model(**sample))
        scores.append(preds.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())
    del sample, label, preds
    scores = np.vstack(scores)
    labels = np.vstack(labels)
    scores = pd.DataFrame(scores)
    labels = pd.DataFrame(labels)
    scores.to_csv(save + 'scores.csv', index=False, header=None)
    labels.to_csv(save + 'labels.csv', index=False, header=None)
    return None
