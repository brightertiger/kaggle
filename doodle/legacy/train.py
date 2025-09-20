import torch
import pickle
import pandas as pd 
import numpy as np 
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader.train_loader import TrainDataset
from dataloader.valid_loader import ValidDataset
from models.resnet18 import ResNet18
from models.resnet34 import ResNet34
from models.resnet50 import ResNet50
from metrics.accuracy import Accuracy
from prettytable import PrettyTable
torch.set_default_tensor_type('torch.DoubleTensor')
np.random.seed(2017)
MODE = 'resnet50'

class ModelTrainer(object):

    def __init__(self, index, size, learning_rate, decay, batches, pretrain=None):
        self.batches = batches
        self.index = index
        self.path = '../data/model/{}/{}_{}.pth'.format(MODE, MODE, self.index)
        self.learning_rate = learning_rate
        self.mapping = pickle.load(open('../data/categories.pkl','rb'))
        self.mapping = [x.replace('.csv','') for x in self.mapping]
        self.train_dataset = '../data/train/train.csv'
        self.valid_dataset = '../data/valid/valid.csv'
        self.train_dataset = pd.read_csv(self.train_dataset, nrows=10000)
        self.valid_dataset = pd.read_csv(self.valid_dataset, nrows=10000)
        self.train_dataset = TrainDataset(self.train_dataset, self.mapping, size)
        self.valid_dataset = ValidDataset(self.valid_dataset, self.mapping, size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=650, shuffle=True, num_workers=11)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=650, shuffle=True, num_workers=11)
        self.model = ResNet50()
        self.model = nn.DataParallel(self.model).float().cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.evaluator = Accuracy(3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if pretrain is not None: self.load_checkpoint(pretrain)
        return None

    def save_checkpoint(self, path):
        state = {'state_dict': self.model.state_dict(), 'optimizer' : self.optimizer.state_dict()}
        torch.save(state, path)
        print('Model Saved.')
        return None
        
    def load_checkpoint(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state['state_dict'])
        print('Model Loaded.')
        return None
    
    def train(self):
        train_loss = []
        train_metric = []
        batch = 0
        for sample in self.train_loader:
            batch += 1
            if batch > self.batches:
                break
            image = Variable(sample['image'].cuda())
            masks = Variable(sample['label'].squeeze().cuda())
            preds = self.model(image)
            loss = self.criterion(preds, masks)
            metric = self.evaluator(preds, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.data)
            train_metric.append(metric)
        return np.mean(train_loss), np.mean(train_metric)
        
    def valid(self):
        valid_loss = []
        valid_metric = []
        batch = 0
        for sample in self.valid_loader:
            batch += 1
            if batch > self.batches:
                break
            image = Variable(sample['image'].cuda())
            masks = Variable(sample['label'].squeeze().cuda())
            preds = self.model(image)
            loss = self.criterion(preds, masks)
            metric = self.evaluator(preds, masks)
            valid_loss.append(loss.data)
            valid_metric.append(metric)
        return np.mean(valid_loss), np.mean(valid_metric)

    def execute(self, checkpoint, epochs):
        counter = 0
        table = PrettyTable(['Source', 'Epoch', 'Rate','Loss','Metric'])
        for epoch in range(epochs):
            counter += 1
            train_loss, train_metric = self.train()
            valid_loss, valid_metric = self.valid()
            table.add_row(['Train', epoch, self.learning_rate, round(train_loss,3), round(train_metric,3)])
            table.add_row(['Valid', epoch, self.learning_rate, round(valid_loss,3), round(valid_metric,3)])
            logger = table.get_string()
            with open('../data/model/{}/{}_{}.log'.format(MODE, MODE, self.learning_rate), 'wb') as f:
                f.write(logger)
            if valid_metric > checkpoint:
                checkpoint = valid_metric
                self.save_checkpoint(self.path)
                counter = 0
            if counter >= 5:
                counter = 0
                self.learning_rate = 0.5 * self.learning_rate
                for param in self.optimizer.param_groups:
                    param['lr'] = self.learning_rate
        return None

