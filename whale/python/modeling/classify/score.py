import torch 
import sys
import numpy as np 
import pandas as pd 
import torch.nn.functional as F 
from torch.autograd import Variable
from tqdm import tqdm 
from utility.checkpoint import load_model
from functools import reduce

def score_model(model, data, batch, path, epochs):
    model, checkpoint = load_model(model, path + 'model.pth')
    model.eval()
    scores = pd.DataFrame([])
    for epoch in range(epochs):
        tq = tqdm(total=len(data) * batch, ncols=0)
        score = []
        index = []
        for sample in data:
            image = Variable(sample['image'].float().cuda())
            name = np.array(sample['idx']).reshape(-1,1)
            preds = F.softmax(model(image), dim=1).cpu().data.numpy()
            score.append(preds)
            index.append(name)
            tq.update(preds.shape[0])
        tq.close()
        score = np.vstack(score)
        index = np.vstack(index)
        index = pd.DataFrame(index, columns=['Image'])
        score = pd.DataFrame(score, columns=['scr_' + str(x) for x in range(5004)])
        score = index.join(score)
        scores = scores.append(score)
        print('Records:', scores.shape[0])
        sys.stdout.flush()
    scores = scores.groupby('Image').mean().reset_index()
    scores.to_csv(path + 'scores.csv', index=False)
    return None
