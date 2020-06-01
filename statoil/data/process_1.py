import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def normalize(array):
    array = (array - array.mean()) / array.std()
    return array

def convert_image(dataframe):
    images =[]
    for i, row in dataframe.iterrows():
        horizontal = np.array(row['band_1']).reshape(75,75)
        vertical = np.array(row['band_2']).reshape(75,75)
        transform1 = np.fabs(np.subtract(vertical,horizontal))
        transform2 = np.maximum(vertical,horizontal)
        transform3 = np.minimum(vertical,horizontal)
        collect = []
        collect += [normalize(transform1)]
        collect += [normalize(transform2)]
        collect += [normalize(transform3)]
        images.append(np.dstack(collect))
    return np.array(images)

data = pd.read_json('../data/download/train.json')
print('data:', data.shape)
images = convert_image(data[['band_1','band_2']])
angles = pd.to_numeric(data['inc_angle'],errors='coerce')
fill_value = np.mean(angles)
angles.fillna(value=fill_value, inplace=True)
min_value = angles.min()
max_value = angles.max()
angles = (angles - min_value) / (max_value - min_value)
labels = np.array(data['is_iceberg'].values)
ids = np.array(data['id'].values)

folds = StratifiedKFold(n_splits=5,random_state=2017, shuffle=True)

suffix = 1

for train, test in folds.split(X=images, y=labels):
    print('fold:', suffix)
    train_images = images[train]
    test_images = images[test]
    train_angles = angles[train]
    test_angles = angles[test]
    train_labels = labels[train]
    test_labels = labels[test]
    train_ids = ids[train]
    test_ids = ids[test]
    print('images:', train_images.shape, test_images.shape)
    print('angles:', train_angles.shape, test_angles.shape)
    print('labels:', train_labels.shape, test_labels.shape)
    print('ids:', train_ids.shape, test_ids.shape)
    np.save('../data/data/source_1/train/train_images_{}'.format(suffix), train_images)
    np.save('../data/data/source_1/train/test_images_{}'.format(suffix), test_images)
    np.save('../data/data/source_1/train/train_labels_{}'.format(suffix), train_labels)
    np.save('../data/data/source_1/train/test_labels_{}'.format(suffix), test_labels)
    np.save('../data/data/source_1/train/train_angles_{}'.format(suffix), train_angles)
    np.save('../data/data/source_1/train/test_angles_{}'.format(suffix), test_angles)
    np.save('../data/data/source_1/train/train_ids_{}'.format(suffix), train_ids)
    np.save('../data/data/source_1/train/test_ids_{}'.format(suffix), test_ids)
    suffix += 1

data = pd.read_json('../data/download/test.json')
print('data:', data.shape)
images = convert_image(data[['band_1','band_2']])
angles = pd.to_numeric(data['inc_angle'],errors='coerce')
angles.fillna(value=fill_value, inplace=True)
angles = (angles - min_value) / (max_value - min_value)
ids = np.array(data['id'].values)
np.save('../data/data/source_1/score/images', images)
np.save('../data/data/source_1/score/angles', angles)
np.save('../data/data/source_1/score/ids', ids)
print('images:', images.shape)
print('angles:', angles.shape)
print('ids:', ids.shape)