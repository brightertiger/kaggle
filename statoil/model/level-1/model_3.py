import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})

np.random.seed(2017)

def define_model(switch, rate):
    # inputs
    input_1 = Input(shape=(75,75,3), name='image')
    input_2 = Input(shape=(1,), name='angle')
    angle = Dense(1,)(input_2)
    vgg = VGG16(input_tensor=input_1, pooling='max', include_top=False)
    for layer in vgg.layers:
        layer.trainable = switch
    convolve = Dropout(0.3)(vgg.output)
    # concatenate
    concat = concatenate([convolve, angle])
    concat = Dense(512, activation='swish', kernel_initializer='he_normal')(concat)
    concat = Dropout(0.2)(concat)
    concat = Dense(256, activation='swish', kernel_initializer='he_normal')(concat)
    concat = Dropout(0.2)(concat)
    predict = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(concat)
    # model
    model = Model(inputs=[input_1, input_2], output=predict)
    optimizer = Adam(lr=rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

params = {}
params['horizontal_flip'] = True
params['vertical_flip'] = True
params['zoom_range'] = 0.3
params['rotation_range'] = 5
params['width_shift_range'] = 0.1
params['height_shift_range'] = 0.1

generator = ImageDataGenerator(**params)

def dataflow(image, angle, label):
    flow_1 = generator.flow(image, label, batch_size=32,seed=2017)
    flow_2 = generator.flow(image, angle, batch_size=32,seed=2017)
    while True:
        tuple_1 = flow_1.next()
        tuple_2 = flow_2.next()
        yield [tuple_1[0], tuple_2[1]], tuple_1[1]

def callbacks(suffix):
    stop = EarlyStopping('val_loss', patience=25, mode="min")
    path = '../data/data/source_1/model_3/model_{}.hdf5'.format(suffix)
    save = ModelCheckpoint(path, save_best_only=True, save_weights_only=True)
    logger = CSVLogger('../data/data/source_1/model_3/logger_{}.log'.format(suffix))
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='min')
    return [stop, save, reduce, logger]

train_image = np.load('../data/data/source_1/train/train_images_1.npy')
train_angle = np.load('../data/data/source_1/train/train_angles_1.npy')
train_label = np.load('../data/data/source_1/train/train_labels_1.npy')
test_image = np.load('../data/data/source_1/train/test_images_1.npy')
test_angle = np.load('../data/data/source_1/train/test_angles_1.npy')
test_label = np.load('../data/data/source_1/train/test_labels_1.npy')
train_generator = dataflow(train_image, train_angle, train_label)
test_generator = ([test_image, test_angle], test_label)

params = {}
params['generator'] = train_generator
params['validation_data'] = test_generator
params['steps_per_epoch'] = 40
params['epochs'] = 100
params['verbose'] = 0
params['callbacks'] = callbacks(1)
model_1 = define_model(False, 1e-4)
model_1.fit_generator(**params)
params['callbacks'] = callbacks(1)
model_1 = define_model(True, 5e-5)
model_1.load_weights('../data/data/source_1/model_3/model_1.hdf5')
model_1.fit_generator(**params)

train_image = np.load('../data/data/source_1/train/train_images_2.npy')
train_angle = np.load('../data/data/source_1/train/train_angles_2.npy')
train_label = np.load('../data/data/source_1/train/train_labels_2.npy')
test_image = np.load('../data/data/source_1/train/test_images_2.npy')
test_angle = np.load('../data/data/source_1/train/test_angles_2.npy')
test_label = np.load('../data/data/source_1/train/test_labels_2.npy')
train_generator = dataflow(train_image, train_angle, train_label)
test_generator = ([test_image, test_angle], test_label)

params = {}
params['generator'] = train_generator
params['validation_data'] = test_generator
params['steps_per_epoch'] = 40
params['epochs'] = 100
params['verbose'] = 0
params['callbacks'] = callbacks(2)
model_1 = define_model(False, 1e-4)
model_1.fit_generator(**params)
params['callbacks'] = callbacks(2)
model_1 = define_model(True, 5e-5)
model_1.load_weights('../data/data/source_1/model_3/model_2.hdf5')
model_1.fit_generator(**params)

train_image = np.load('../data/data/source_1/train/train_images_3.npy')
train_angle = np.load('../data/data/source_1/train/train_angles_3.npy')
train_label = np.load('../data/data/source_1/train/train_labels_3.npy')
test_image = np.load('../data/data/source_1/train/test_images_3.npy')
test_angle = np.load('../data/data/source_1/train/test_angles_3.npy')
test_label = np.load('../data/data/source_1/train/test_labels_3.npy')
train_generator = dataflow(train_image, train_angle, train_label)
test_generator = ([test_image, test_angle], test_label)

params = {}
params['generator'] = train_generator
params['validation_data'] = test_generator
params['steps_per_epoch'] = 40
params['epochs'] = 100
params['verbose'] = 0
params['callbacks'] = callbacks(3)
model_1 = define_model(False, 1e-4)
model_1.fit_generator(**params)
params['callbacks'] = callbacks(3)
model_1 = define_model(True, 5e-5)
model_1.load_weights('../data/data/source_1/model_3/model_3.hdf5')
model_1.fit_generator(**params)

train_image = np.load('../data/data/source_1/train/train_images_4.npy')
train_angle = np.load('../data/data/source_1/train/train_angles_4.npy')
train_label = np.load('../data/data/source_1/train/train_labels_4.npy')
test_image = np.load('../data/data/source_1/train/test_images_4.npy')
test_angle = np.load('../data/data/source_1/train/test_angles_4.npy')
test_label = np.load('../data/data/source_1/train/test_labels_4.npy')
train_generator = dataflow(train_image, train_angle, train_label)
test_generator = ([test_image, test_angle], test_label)

params = {}
params['generator'] = train_generator
params['validation_data'] = test_generator
params['steps_per_epoch'] = 40
params['epochs'] = 100
params['verbose'] = 0
params['callbacks'] = callbacks(4)
model_1 = define_model(False, 1e-4)
model_1.fit_generator(**params)
params['callbacks'] = callbacks(4)
model_1 = define_model(True, 5e-5)
model_1.load_weights('../data/data/source_1/model_3/model_4.hdf5')
model_1.fit_generator(**params)

train_image = np.load('../data/data/source_1/train/train_images_5.npy')
train_angle = np.load('../data/data/source_1/train/train_angles_5.npy')
train_label = np.load('../data/data/source_1/train/train_labels_5.npy')
test_image = np.load('../data/data/source_1/train/test_images_5.npy')
test_angle = np.load('../data/data/source_1/train/test_angles_5.npy')
test_label = np.load('../data/data/source_1/train/test_labels_5.npy')
train_generator = dataflow(train_image, train_angle, train_label)
test_generator = ([test_image, test_angle], test_label)

params = {}
params['generator'] = train_generator
params['validation_data'] = test_generator
params['steps_per_epoch'] = 40
params['epochs'] = 100
params['verbose'] = 0
params['callbacks'] = callbacks(5)
model_1 = define_model(False, 1e-4)
model_1.fit_generator(**params)
params['callbacks'] = callbacks(5)
model_1 = define_model(True, 5e-5)
model_1.load_weights('../data/data/source_1/model_3/model_5.hdf5')
model_1.fit_generator(**params)