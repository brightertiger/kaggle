import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import os

class DataProcessor:
    def __init__(self, config):
        self.config = config
        
    def normalize_array(self, array):
        return (array - array.mean()) / array.std()
    
    def convert_images_source1(self, dataframe):
        images = []
        for i, row in dataframe.iterrows():
            horizontal = np.array(row['band_1']).reshape(75, 75)
            vertical = np.array(row['band_2']).reshape(75, 75)
            
            transform1 = np.fabs(np.subtract(vertical, horizontal))
            transform2 = np.maximum(vertical, horizontal)
            transform3 = np.minimum(vertical, horizontal)
            
            collect = []
            collect += [self.normalize_array(transform1)]
            collect += [self.normalize_array(transform2)]
            collect += [self.normalize_array(transform3)]
            images.append(np.dstack(collect))
        return np.array(images)
    
    def convert_images_source2(self, dataframe):
        images = []
        for i, row in dataframe.iterrows():
            horizontal = np.array(row['band_1']).reshape(75, 75)
            vertical = np.array(row['band_2']).reshape(75, 75)
            transform = (vertical + horizontal) / 2
            
            collect = []
            collect += [self.normalize_array(vertical)]
            collect += [self.normalize_array(horizontal)]
            collect += [self.normalize_array(transform)]
            images.append(np.dstack(collect))
        return np.array(images)
    
    def process_angles(self, angles):
        angles = pd.to_numeric(angles, errors='coerce')
        fill_value = np.mean(angles)
        angles.fillna(value=fill_value, inplace=True)
        
        min_value = angles.min()
        max_value = angles.max()
        angles = (angles - min_value) / (max_value - min_value)
        return angles
    
    def create_folds(self, images, labels, angles, ids, source_name):
        folds = StratifiedKFold(n_splits=self.config.FOLDS, 
                              random_state=self.config.RANDOM_STATE, 
                              shuffle=True)
        
        os.makedirs(f'{self.config.DATA_DIR}/{source_name}/train', exist_ok=True)
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds.split(X=images, y=labels), 1):
            train_images = images[train_idx]
            test_images = images[test_idx]
            train_angles = angles[train_idx]
            test_angles = angles[test_idx]
            train_labels = labels[train_idx]
            test_labels = labels[test_idx]
            train_ids = ids[train_idx]
            test_ids = ids[test_idx]
            
            np.save(f'{self.config.DATA_DIR}/{source_name}/train/train_images_{fold_idx}', train_images)
            np.save(f'{self.config.DATA_DIR}/{source_name}/train/test_images_{fold_idx}', test_images)
            np.save(f'{self.config.DATA_DIR}/{source_name}/train/train_labels_{fold_idx}', train_labels)
            np.save(f'{self.config.DATA_DIR}/{source_name}/train/test_labels_{fold_idx}', test_labels)
            np.save(f'{self.config.DATA_DIR}/{source_name}/train/train_angles_{fold_idx}', train_angles)
            np.save(f'{self.config.DATA_DIR}/{source_name}/train/test_angles_{fold_idx}', test_angles)
            np.save(f'{self.config.DATA_DIR}/{source_name}/train/train_ids_{fold_idx}', train_ids)
            np.save(f'{self.config.DATA_DIR}/{source_name}/train/test_ids_{fold_idx}', test_ids)
    
    def process_train_data(self, source_name, convert_func):
        data = pd.read_json(f'{self.config.DATA_DIR}/download/train.json')
        images = convert_func(data[['band_1', 'band_2']])
        angles = self.process_angles(data['inc_angle'])
        labels = np.array(data['is_iceberg'].values)
        ids = np.array(data['id'].values)
        
        self.create_folds(images, labels, angles, ids, source_name)
        return images, angles, labels, ids
    
    def process_test_data(self, source_name, convert_func):
        data = pd.read_json(f'{self.config.DATA_DIR}/download/test.json')
        images = convert_func(data[['band_1', 'band_2']])
        angles = self.process_angles(data['inc_angle'])
        ids = np.array(data['id'].values)
        
        os.makedirs(f'{self.config.DATA_DIR}/{source_name}/score', exist_ok=True)
        np.save(f'{self.config.DATA_DIR}/{source_name}/score/images', images)
        np.save(f'{self.config.DATA_DIR}/{source_name}/score/angles', angles)
        np.save(f'{self.config.DATA_DIR}/{source_name}/score/ids', ids)
        
        return images, angles, ids
