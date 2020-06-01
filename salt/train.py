import pandas as pd 
import numpy as np 
import torch
import pickle
from torch import nn
from torch.autograd import Variable
from dataloaders.dataloader import ImageDataset
from models.seresnet34 import UNetResNet34
from metrics.dice import DiceLoss 
from metrics.iou import IOUMetric
from metrics.lovasz import LovaszLoss
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.DoubleTensor')
np.random.seed(2017)

class ModelTrainer(object):

    def __init__(self, fold):
        self.fold = fold
        self.train_idx = pickle.load(open('../data/data/fold_{}/train/train.pkl'.format(fold),'rb'))
        self.valid_idx = pickle.load(open('../data/data/fold_{}/valid/valid.pkl'.format(fold),'rb'))
        self.train_path = '../data/data/fold_{}/train/'.format(fold)
        self.valid_path = '../data/data/fold_{}/valid/'.format(fold)
        self.train_dataset = ImageDataset(self.train_idx, self.train_path, flip=True)
        self.valid_dataset = ImageDataset(self.valid_idx, self.valid_path, flip=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=32, shuffle=True, num_workers=4)
        return None

    def save_checkpoint(self, path, model, optimizer):
        state = {'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
        torch.save(state, path)
        print('Model Saved.')
        return None
        
    def load_checkpoint(self, path, model):
        state = torch.load(path)
        model.load_state_dict(state['state_dict'])
        print('Model Loaded.')
        return None
    
    def cosine_rate(self, iteration):
        fetch = iteration % 20
        rates = list(np.arange(0.00001,0.001,0.00005))
        rates = list(reversed(rates))
        return rates[fetch]

    def train(self, metric, criterion, evaluator, epochs, learning_rate, decay, pretrain):
        path = '../data/data/model/model_{}.pth'.format(self.fold)
        model = UNetResNet34(pretrained=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
        best_iteration = metric
        counter = 0
        if pretrain: self.load_checkpoint(path, model)
        for epoch in range(epochs):
            print('Epoch:', epoch)
            print('Learning Rate:', learning_rate)
            train_loss = []
            valid_loss = []
            train_metric = []
            valid_metric = []
            for sample in self.train_loader:
                model = model.float().cuda()
                image = Variable(sample['image'].cuda())
                masks = Variable(sample['mask'].cuda())
                preds = model(image)
                loss = criterion(preds, masks)
                metric = evaluator(preds, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.data)
                train_metric.append(metric) 
            for sample in self.valid_loader:
                image = Variable(sample['image'].cuda())
                masks = Variable(sample['mask'].cuda())
                preds = model(image)
                loss = criterion(preds, masks)
                metric = evaluator(preds, masks)
                valid_loss.append(loss.data)
                valid_metric.append(metric)
            train_loss = np.mean(train_loss)
            valid_loss = np.mean(valid_loss) 
            train_metric = np.mean(train_metric)
            valid_metric = np.mean(valid_metric)
            print("Iteration: %d, Train Loss: %.3f, Valid Loss: %.3f" % (epoch, train_loss, valid_loss))
            print("Iteration: %d, Train Metric: %.3f, Valid Metric: %.3f" % (epoch, train_metric, valid_metric))
            last_iteration = valid_metric
            if valid_metric > best_iteration: 
                self.save_checkpoint(path, model, optimizer)
                best_iteration = valid_metric
                counter = 0
            if valid_metric < best_iteration: 
                counter += 1
                print("Model hasn't improved in last {} iterations.".format(counter))
            if counter >= 50:
                print("Stopping Training.")
                break
            if counter >=10:
                counter = 0
                learning_rate = 0.5 * learning_rate
                for param in optimizer.param_groups:
                    param['lr'] = learning_rate
        return None

if __name__ == '__main__':
    FOLD = 1
    execute = ModelTrainer(FOLD)
    params = {}
    params['metric'] = 0.78
    #params['criterion'] = DiceLoss(bce_weight=1.0, dice_weight=1.0)
    params['criterion'] = LovaszLoss()
    params['evaluator'] = IOUMetric(cutoff=0., squash=False)
    params['epochs'] = 200
    params['learning_rate'] = 0.001
    params['decay'] = 0.0001
    params['pretrain'] = True
    execute.train(**params)


