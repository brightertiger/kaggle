import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn import functional as F
from torch.autograd import Variable

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def loadModel(model, path):
    results = torch.load(path)
    model.load_state_dict(results['model_state_dict'])
    loss = results['loss']
    print('Model Loaded:', 'Loss:', loss)
    return model, loss

def scoreModel(model, data, load_path, save_path):
    model.eval()
    model, _ = loadModel(model, load_path)
    idx = []
    scores = []
    for sample in data:
        idx += sample['idx']
        image = Variable(sample['image'].float().to(DEVICE))
        preds = torch.sigmoid(model(image))
        scores += [preds.cpu().data.numpy()]
    scores = np.vstack(scores)
    index = np.vstack(idx)
    index = pd.DataFrame(index, columns=['ID'])
    scores = pd.DataFrame(np.array(scores))
    scores.columns = ['scr_' + str(x) for x in range(scores.shape[1])]
    scores = index.join(scores)
    scores.to_csv(save_path, index=False)
    print('Score Data:', scores.shape)
    return None