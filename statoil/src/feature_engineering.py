import pandas as pd
import numpy as np
import cv2
from scipy.stats import kurtosis, skew
from scipy.ndimage import laplace, sobel
from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm
import gc

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        
    def read_json_data(self, file_path):
        df = pd.read_json(file_path)
        df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
        
        band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
        band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
        
        df = df.drop(['band_1', 'band_2'], axis=1)
        bands = np.stack((band1, band2, 0.5 * (band1 + band2)), axis=-1)
        
        return df, bands
    
    def extract_image_statistics(self, img_data):
        img_id, img = img_data[0], img_data[1]
        np.seterr(divide='ignore', invalid='ignore')
        
        bins = 20
        scl_min, scl_max = -50, 50
        opt_poly = True
        
        features = []
        features_interv = []
        hist_interv = []
        
        for i in range(img.shape[2]):
            img_sub = np.squeeze(img[:, :, i])
            sub_features = []
            
            sub_features += [np.mean(img_sub), np.std(img_sub), np.max(img_sub), 
                           np.median(img_sub), np.min(img_sub)]
            sub_features += [(sub_features[2] - sub_features[3]), 
                           (sub_features[2] - sub_features[4]), 
                           (sub_features[3] - sub_features[4])]
            sub_features += [(sub_features[-3] / sub_features[1]), 
                           (sub_features[-2] / sub_features[1]), 
                           (sub_features[-1] / sub_features[1])]
            
            features += sub_features
            
            transform_features = []
            transform_features += [laplace(img_sub, mode='reflect', cval=0.0).ravel().var()]
            
            sobel0 = sobel(img_sub, axis=0, mode='reflect', cval=0.0).ravel().var()
            sobel1 = sobel(img_sub, axis=1, mode='reflect', cval=0.0).ravel().var()
            transform_features += [sobel0, sobel1]
            transform_features += [kurtosis(img_sub.ravel()), skew(img_sub.ravel())]
            
            if opt_poly:
                features_interv.append(sub_features)
                features += [x * y for x, y in combinations(transform_features, 2)]
                features += [x + y for x, y in combinations(transform_features, 2)]
                features += [x - y for x, y in combinations(transform_features, 2)]
            
            hist = list(np.histogram(img_sub, bins=bins, range=(scl_min, scl_max))[0])
            hist_interv.append(hist)
            features += hist
            features += [hist.index(max(hist))]
            features += [np.std(hist), np.max(hist), np.median(hist), 
                        (np.max(hist) - np.median(hist))]
        
        if opt_poly:
            for x, y in combinations(features_interv, 2):
                features += [float(x[j]) * float(y[j]) for j in range(len(features_interv[0]))]
            
            for x, y in combinations(hist_interv, 2):
                hist_diff = [x[j] * y[j] for j in range(len(hist_interv[0]))]
                features += [hist_diff.index(max(hist_diff))]
                features += [np.std(hist_diff), np.max(hist_diff)]
                features += [np.median(hist_diff), (np.max(hist_diff) - np.median(hist_diff))]
        
        nan_value = -999
        for i in range(len(features)):
            if np.isnan(features[i]):
                features[i] = nan_value
        
        return [img_id, features]
    
    def extract_features_parallel(self, img_data_list):
        feature_dict = {}
        p = Pool(2)
        results = p.map(self.extract_image_statistics, img_data_list)
        
        for i in tqdm(range(len(results)), miniters=100):
            feature_dict[results[i][0]] = results[i][1]
        
        results = []
        feature_data = [feature_dict[i] for i, j in img_data_list]
        return np.array(feature_data, dtype=np.float32)
    
    def process_features(self, df, bands):
        data = self.extract_features_parallel([(k, v) for k, v in zip(df['id'].tolist(), bands)])
        gc.collect()
        
        data = np.concatenate([data, df['inc_angle'].values[:, np.newaxis]], axis=-1)
        gc.collect()
        
        return data
    
    def create_xgboost_features(self, train_file, test_file):
        selected_features = [246, 46, 169, 35, 163, 99, 153, 170, 34, 38]
        feature_names = [f'feat_{i}' for i in range(len(selected_features))]
        
        train_df, train_bands = self.read_json_data(train_file)
        X_train = self.process_features(df=train_df, bands=train_bands)
        X_train = pd.DataFrame(X_train)[selected_features]
        X_train.columns = feature_names
        train_output = train_df[['id', 'inc_angle']].join(X_train)
        train_output.to_csv(f'{self.config.DATA_DIR}/train_xgb.csv', index=False)
        
        test_df, test_bands = self.read_json_data(test_file)
        X_test = self.process_features(df=test_df, bands=test_bands)
        X_test = pd.DataFrame(X_test)[selected_features]
        X_test.columns = feature_names
        test_output = test_df[['id', 'inc_angle']].join(X_test)
        test_output.to_csv(f'{self.config.DATA_DIR}/test_xgb.csv', index=False)
        
        return train_output, test_output
