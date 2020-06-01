import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import log_loss
from functools import reduce

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})

np.random.seed(2017)

def define_model():
    # inputs
    input_1 = Input(shape=(75,75,3), name='image')
    input_2 = Input(shape=(1,), name='angle')
    angle = Dense(1,)(input_2)
    # convolution
    convolve = Conv2D(64, kernel_size=(3, 3), padding='same')(input_1)
    convolve = BatchNormalization()(convolve)
    convolve = Activation('swish')(convolve)
    convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
    convolve = Conv2D(128, kernel_size=(3, 3), padding='same')(convolve)
    convolve = BatchNormalization()(convolve)
    convolve = Activation('swish')(convolve)
    convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
    convolve = Conv2D(256, kernel_size=(3, 3), padding='same')(convolve)
    convolve = BatchNormalization()(convolve)
    convolve = Activation('swish')(convolve)
    convolve = Conv2D(256, kernel_size=(3, 3), padding='same')(convolve)
    convolve = BatchNormalization()(convolve)
    convolve = Activation('swish')(convolve)
    convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
    convolve = Conv2D(512, kernel_size=(3, 3), padding='same')(convolve)
    convolve = BatchNormalization()(convolve)
    convolve = Activation('swish')(convolve)
    convolve = Conv2D(512, kernel_size=(3, 3), padding='same')(convolve)
    convolve = BatchNormalization()(convolve)
    convolve = Activation('swish')(convolve)
    convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
    convolve = Flatten()(convolve)
    convolve = Dropout(0.3)(convolve)
    # concatenate
    concat = concatenate([convolve, angle])
    concat = Dense(512, activation='swish', kernel_initializer='he_normal')(concat)
    concat = Dropout(0.3)(concat)
    concat = Dense(256, activation='swish', kernel_initializer='he_normal')(concat)
    concat = Dropout(0.3)(concat)
    predict = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(concat)
    # model
    model = Model(inputs=[input_1, input_2], output=predict)
    optimizer = Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

test_image = np.load('../data/data/source_1/score/images.npy')
test_angle = np.load('../data/data/source_1/score/angles.npy')
test_ids = np.load('../data/data/source_1/score/ids.npy')
test_generator = [test_image, test_angle]

def score(suffix):
    model = define_model()
    path = '../data/data/source_1/model_1/model_{}.hdf5'.format(suffix)
    model.load_weights(path)
    score = model.predict(test_generator)
    K.clear_session()
    return score

model_1 = pd.Series(score(1)[:,0], name='model_1')
model_2 = pd.Series(score(2)[:,0], name='model_2')
model_3 = pd.Series(score(3)[:,0], name='model_3')
model_4 = pd.Series(score(4)[:,0], name='model_4')
model_5 = pd.Series(score(5)[:,0], name='model_5')
ids = pd.Series(test_ids, name='id')
scores = pd.DataFrame([ids, model_1, model_2, model_3, model_4, model_5])
scores = scores.T
scores['is_iceberg'] = scores.iloc[:,1:].mean(axis=1)
scores = scores[['id','is_iceberg']]
print('score file:', scores.shape)

scores.to_csv('../data/submit/baseline_v1.csv', index=False)

def score(suffix):
    test_image = np.load('../data/data/source_1/train/test_images_{}.npy'.format(suffix))
    test_angle = np.load('../data/data/source_1/train/test_angles_{}.npy'.format(suffix))
    test_ids = np.load('../data/data/source_1/train/test_ids_{}.npy'.format(suffix))
    test_labels = np.load('../data/data/source_1/train/test_labels_{}.npy'.format(suffix))
    test_generator = [test_image, test_angle]
    model = define_model()
    path = '../data/data/source_1/model_1/model_{}.hdf5'.format(suffix)
    model.load_weights(path)
    data = pd.DataFrame(model.predict(test_generator)[:,0], columns=['score'])
    K.clear_session()
    data['id'] = test_ids
    data['label'] = test_labels
    data['score'] = np.clip(data['score'], 0.0001,0.9999)
    print('log loss:', log_loss(data['label'], data['score']))
    return data[['id','label','score']]

crossfolds = []
for i in range(5):
    crossfolds.append(score(i+1))
    
crossfolds = reduce(lambda x,y : x.append(y), crossfolds)
print('log loss:', log_loss(crossfolds['label'], crossfolds['score']))
crossfolds.to_csv('../data/model/baseline_v1.csv', index=False)