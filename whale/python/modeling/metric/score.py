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
    embeddings = pd.DataFrame([])
    for epoch in range(epochs):
        tq = tqdm(total=len(data) * batch, ncols=0)
        score = []
        embedding = []
        index = []
        for sample in data:
            image = Variable(sample['image'].float().cuda())
            name = np.array(sample['idx']).reshape(-1,1)
            preds, embed = model(image) 
            preds = preds.cpu()
            preds = F.softmax(preds, dim=1).data.numpy()
            embed = embed.cpu().data.numpy()
            score.append(preds)
            embedding.append(embed)
            index.append(name)
            tq.update(preds.shape[0])
        tq.close()
        score = np.vstack(score)
        embedding = np.vstack(embedding)
        index = np.vstack(index)
        index = pd.DataFrame(index, columns=['Image'])
        score = pd.DataFrame(score, columns=['scr_' + str(x) for x in range(5004)])
        embedding = pd.DataFrame(embedding, columns=['embd_' + str(x) for x in range(256)])
        score = index.join(score)
        embedding = index.join(embedding)
        scores = scores.append(score)
        embeddings = embeddings.append(embedding)
        print('Records:', scores.shape[0], embeddings.shape[0])
        sys.stdout.flush()
    scores = scores.groupby('Image').mean().reset_index()
    embeddings = embeddings.groupby('Image').mean().reset_index()
    scores.to_csv(path + 'scores.csv', index=False)
    embeddings.to_csv(path + 'embeddings.csv', index=False)
    return None
