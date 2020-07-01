import torch 
import torch.nn as nn
import pandas as pd
import numpy as np

def scoreModel(model, data, save):
    model.eval()
    start_idxs = []
    end_idxs = []
    start_preds = []
    end_preds = []
    for sample in data:
        for key, value in sample.items(): sample[key] = value.cuda() 
        s_pred, e_pred, _ = model(sample['tokens'], sample['masks'])
        s_pred, e_pred = s_pred.argmax(dim=1), e_pred.argmax(dim=1)
        s_pred, e_pred = s_pred.data.cpu().numpy(), e_pred.data.cpu().numpy()
        s_tok, e_tok, _ = sample['start_idx'], sample['end_idx'], sample['aux_label']
        s_tok, e_tok = s_tok.data.cpu().numpy(), e_tok.data.cpu().numpy()
        start_idxs.append(s_tok)
        end_idxs.append(e_tok)
        start_preds.append(s_pred)
        end_preds.append(e_pred)
    del sample
    start_idxs = np.hstack(start_idxs)
    end_idxs = np.hstack(end_idxs)
    start_preds = np.hstack(start_preds)
    end_preds = np.hstack(end_preds)
    scores = pd.DataFrame(zip(start_idxs, end_idxs, start_preds, end_preds))
    scores.columns = ['start_idx','end_idx', 'start_pred', 'end_pred']
    print(scores.shape)
    scores.to_csv(save, index=False)
    return None