import torch
import pickle
import pandas as pd
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader.score_loader import ScoreDataset
from models.resnet34 import ResNet34
from metrics.precision import Accuracy
torch.set_default_tensor_type('torch.DoubleTensor')
np.random.seed(2017)
MODE = 'mobilenet'

class ModelScorer(object):

    def __init__(self, model_path, score_path):
        self.model_path = model_path
        self.score_path = score_path
        self.mapping = pickle.load(open('../data/categories.pkl','rb'))
        self.mapping = [x.replace('.csv','') for x in self.mapping]
        self.score_dataset = '../data/test/test_simplified.csv'
        self.score_dataset = pd.read_csv(self.score_dataset)
        self.key_mapping = self.score_dataset.key_id.tolist()
        self.score_dataset = ScoreDataset(self.score_dataset, self.mapping, size=64)
        self.score_loader = DataLoader(self.score_dataset, batch_size=6000, shuffle=False, num_workers=7)
        self.model = ResNet34()
        self.load_checkpoint(self.model_path)
        self.model = self.model.float().cuda()
        return None

    def load_checkpoint(self, path):
        state = torch.load(path)
        weights = {}
        for key, value in state['state_dict'].items():
            weights[key.replace('module.','')] = value    
        self.model.load_state_dict(weights)
        print('Model Loaded.')
        return None
    
    def execute(self):
        scores = []
        for sample in self.score_loader:
            image = Variable(sample['image'].cuda())
            preds = self.model(image).cpu().data.numpy()
            scores.append(preds)
        scores = np.vstack(scores)
        scores = pd.DataFrame(scores)
        scores['key_id'] = self.key_mapping
        scores.to_csv(self.score_path, index=False)
        return None 

if __name__ == '__main__':
    build = ModelScorer('../data/model/resnet18/mobilenet_4.pth', '../data/score/mobilenet/mobilenet_4.csv')
    build.execute()