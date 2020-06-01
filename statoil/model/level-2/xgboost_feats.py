import pandas as pd
import numpy as np
import xgboost as xgb
import cv2
from random import choice, sample, shuffle, uniform, seed
from math import exp, expm1, log1p, log10, log2, sqrt, ceil, floor, isfinite, isnan
from itertools import combinations
from scipy.stats import kurtosis, skew
from scipy.ndimage import laplace, sobel
from sklearn.metrics import log_loss
from tqdm import tqdm
from multiprocessing import Pool
import gc

def read_json(file):
    df = pd.read_json(file)
    df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)
    bands = np.stack((band1, band2,  0.5 * (band1 + band2)), axis=-1)
    return df, bands

def img_to_stats(paths):
    img_id, img = paths[0], paths[1]
    np.seterr(divide='ignore', invalid='ignore')
    bins = 20
    scl_min, scl_max = -50, 50
    opt_poly = True
    st = []
    st_interv = []
    hist_interv = []
    for i in range(img.shape[2]):
        img_sub = np.squeeze(img[:, :, i])
        sub_st = []
        sub_st += [np.mean(img_sub), np.std(img_sub), np.max(img_sub), np.median(img_sub), np.min(img_sub)]
        sub_st += [(sub_st[2] - sub_st[3]), (sub_st[2] - sub_st[4]), (sub_st[3] - sub_st[4])] 
        sub_st += [(sub_st[-3] / sub_st[1]), (sub_st[-2] / sub_st[1]), (sub_st[-1] / sub_st[1])]
        st += sub_st
        st_trans = []
        st_trans += [laplace(img_sub, mode='reflect', cval=0.0).ravel().var()] #blurr
        sobel0 = sobel(img_sub, axis=0, mode='reflect', cval=0.0).ravel().var()
        sobel1 = sobel(img_sub, axis=1, mode='reflect', cval=0.0).ravel().var()
        st_trans += [sobel0, sobel1]
        st_trans += [kurtosis(img_sub.ravel()), skew(img_sub.ravel())]
        if opt_poly:
            st_interv.append(sub_st)
            st += [x * y for x, y in combinations(st_trans, 2)]
            st += [x + y for x, y in combinations(st_trans, 2)]
            st += [x - y for x, y in combinations(st_trans, 2)]                
        hist = list(np.histogram(img_sub, bins=bins, range=(scl_min, scl_max))[0])
        hist_interv.append(hist)
        st += hist
        st += [hist.index(max(hist))]
        st += [np.std(hist), np.max(hist), np.median(hist), (np.max(hist) - np.median(hist))]
    if opt_poly:
        for x, y in combinations(st_interv, 2):
            st += [float(x[j]) * float(y[j]) for j in range(len(st_interv[0]))]
        for x, y in combinations(hist_interv, 2):
            hist_diff = [x[j] * y[j] for j in range(len(hist_interv[0]))]
            st += [hist_diff.index(max(hist_diff))]
            st += [np.std(hist_diff), np.max(hist_diff)]
            st += [np.median(hist_diff),(np.max(hist_diff) - np.median(hist_diff))]
    nan = -999
    for i in range(len(st)):
        if isnan(st[i]) == True:
            st[i] = nan
    return [img_id, st]

def extract_img_stats(paths):
    imf_d = {}
    p = Pool(2)
    ret = p.map(img_to_stats, paths)
    for i in tqdm(range(len(ret)), miniters=100):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[i] for i, j in paths]
    return np.array(fdata, dtype=np.float32)

def process(df, bands):
    data = extract_img_stats([(k, v) for k, v in zip(df['id'].tolist(), bands)]); gc.collect()
    data = np.concatenate([data, df['inc_angle'].values[:, np.newaxis]], axis=-1); gc.collect()
    print(data.shape)
    return data

columns = [246,46,169,35,163,99,153,170,34,38]
names = ['feat_' + str(x) for x in range(len(columns))]

train, train_bands = read_json('../data/download/train.json')
X_train = process(df=train, bands=train_bands)
X_train = pd.DataFrame(X_train)[columns]
X_train.columns = names
train = train[['id','inc_angle']].join(X_train)
train.to_csv('../data/train_xgb.csv', index=False)

test, test_bands = read_json('../data/download/test.json')
X_test = process(df=test, bands=test_bands)
X_test = pd.DataFrame(X_test)[columns]
X_test.columns = names
test = test[['id','inc_angle']].join(X_test)
test.to_csv('../data/test_xgb.csv', index=False)