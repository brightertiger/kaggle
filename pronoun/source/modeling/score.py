import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from source.utility.checkpoint import loadModel

def scoreModel(device, model, data, batch, path):
    model, checkpoint = loadModel(model, path + 'model.pth')
    model.eval()
    tq = tqdm(total=len(data) * batch, ncols=0)
    score = []
    for sample in data:
        text = Variable(sample[0].to(device))
        offset = Variable(sample[1].to(device))
        feature_a = Variable(sample[2].to(device))
        feature_b = Variable(sample[3].to(device))
        preds = F.softmax(model(text, offset, feature_a, feature_b), dim=1).cpu().data.numpy()
        score.append(preds)
        tq.update(preds.shape[0])
    tq.close()
    score = np.vstack(score)
    score = pd.DataFrame(score, columns=['A','B','NEITHER'])
    score.to_csv(path + 'scores.csv', index=False)
    return None