import torch
import torchvision
import pickle
import math
import cv2
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torchvision import models
from dataloaders.dataloader import ImageDataset, ScoreDataset
from models.vgg11 import UNetVGG11
from models.resnet34 import UNetResNet34
torch.set_default_tensor_type('torch.DoubleTensor')

FOLD = 1
VALID_IDX = pickle.load(open('../data/data/fold_{}/valid/valid.pkl'.format(FOLD),'rb'))
SCORE_IDX = pickle.load(open('../data/data/test/test.pkl','rb'))
VALID_PATH = '../data/data/fold_{}/valid/'.format(FOLD)
SCORE_PATH = '../data/data/test/'

def load_checkpoint(path, model, optimizer):
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    print('Model Loaded.')

valid_dataset = ImageDataset(VALID_IDX, VALID_PATH, flip=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
score_dataset1 = ScoreDataset(SCORE_IDX, SCORE_PATH, flip=False)
score_dataset2 = ScoreDataset(SCORE_IDX, SCORE_PATH, flip=True)
score_loader1 = DataLoader(score_dataset1, batch_size=32, shuffle=False, num_workers=4)
score_loader2 = DataLoader(score_dataset2, batch_size=32, shuffle=False, num_workers=4)

model = UNetResNet34()
optimizer = torch.optim.Adam(model.parameters())
load_checkpoint('../data/data/model/model_{}.pth'.format(FOLD), model, optimizer)
model = model.float().cuda()
sigmoid = nn.Sigmoid()
valid_scores = []
valid_actuals = []

for sample in valid_loader:
    image = Variable(sample['image'].cuda())
    mask = sample['mask']
    preds = model(image)
    # preds = sigmoid(preds)
    preds = preds.cpu().data.numpy()
    mask = mask.cpu().data.numpy()
    valid_scores += [preds]
    valid_actuals += [mask]

valid_scores = np.vstack(valid_scores)
valid_actuals = np.vstack(valid_actuals)
print('Valid Scoring:', valid_scores.shape, valid_actuals.shape)
np.save('../data/data/scores/valid/scores_{}.npy'.format(FOLD), valid_scores)
np.save('../data/data/scores/valid/actuals_{}.npy'.format(FOLD), valid_actuals)

test_scores1 = []
test_scores2 = []

for sample in score_loader1:
    image = Variable(sample['image'].cuda())
    preds = model(image)
    # preds = sigmoid(preds)
    preds = preds.cpu().data.numpy()
    test_scores1 += [preds]

for sample in score_loader2:
    image = Variable(sample['image'].cuda())
    preds = model(image)
    # preds = sigmoid(preds)
    preds = preds.cpu().data.numpy()
    test_scores2 += [preds]

test_scores1 = np.vstack(test_scores1)
test_scores2 = np.vstack(test_scores2)

test_scores = []
for idx in range(test_scores1.shape[0]):
    preds1 = test_scores1[idx,:,:,:].reshape(128,128,1)
    preds2 = test_scores2[idx,:,:,:].reshape(128,128,1)
    preds2 = np.fliplr(preds2)
    preds = 0.5 * preds1 + 0.5 * preds2
    preds = preds.reshape(1,128,128)
    test_scores.append(preds)

test_scores = np.vstack(test_scores)
print('Test Scoring:', test_scores.shape)
np.save('../data/data/scores/test/scores_{}.npy'.format(FOLD), test_scores)