import sys
sys.path.insert(0,'..')
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F
from apex import amp

DEVICE = 'cuda:0'

def scoreModel(model, data):
    model.eval()
    index_array = []
    pred_array = []
    for sample in data:
        index = sample['idx']
        image = Variable(sample['image'].float().to(DEVICE))
        preds = torch.sigmoid(model(image))
        index_array.extend(index)
        pred_array.append(preds.cpu().data.numpy())
    del image, preds
    torch.cuda.empty_cache()
    pred_array = np.concatenate(pred_array, axis=0)
    data = pd.DataFrame(index_array, columns=['image'])
    score = pd.DataFrame(pred_array, columns=['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural'])
    data = data.join(score)
    return data