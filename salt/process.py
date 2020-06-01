import pandas as pd
import numpy as np 
import cv2
import pickle
import torch
from metrics.iou import IOUMetric
from functools import reduce

RANGE = 2

def rle_encode(im):
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def evaluate(value):
    metric = []
    iou = IOUMetric(value,False)
    for fold in range(1,RANGE):
        scores = np.load('../data/data/scores/valid/scores_{}.npy'.format(fold))
        actuals = np.load('../data/data/scores/valid/actuals_{}.npy'.format(fold))
        scores = torch.from_numpy(scores)
        actuals = torch.from_numpy(actuals)
        metric += [round(iou(scores, actuals), 4)]
    return metric, np.mean(metric)

best_threshold = 0.
best_folds = None
best_value = 0.
for threshold in range(-25,25):
    threshold = threshold / 100
    folds, output = evaluate(threshold)
    if output > best_value:
        best_value = output
        best_folds = folds
        best_threshold = threshold

print('cutoff:', best_threshold, round(best_value,4), best_folds)
# cutoff: -0.15 0.8112 [0.8218, 0.7906, 0.8106, 0.8196, 0.8134]
# cutoff: -0.13 0.8209 [0.8268, 0.821, 0.8351, 0.814, 0.8076]

CUTOFF = -0.18

output = {}
test = pickle.load(open('../data/data/test/test.pkl','rb'))
scores = []
for fold in range(1,RANGE):
    scores.append(np.load('../data/data/scores/test/scores_{}.npy'.format(fold)))

for idx, image in enumerate(test):
    predict = []
    for fold in range(1,RANGE):
        predict.append(scores[fold-1][idx,14:-13,14:-13])
    predict = reduce(lambda x,y : x + y, predict)
    predict = predict / (RANGE-1)
    predict = predict[:,:, np.newaxis]
    predict = predict.reshape(101,101,1)
    predict = np.int32(predict >= CUTOFF)
    if predict.sum() <= 25:
        predict = np.zeros_like(predict)
    predict = rle_encode(predict)
    output[image] = predict

submit = pd.DataFrame.from_dict(list(output.items()))
submit.columns = ['id','rle_mask']
submit.to_csv('../data/data/submit/seresnet34.csv', index=False)
