import pandas as pd
import numpy as np
import cv2
from keras.layers import Dense, Dropout, Embedding
from keras.layers import Input, Activation, concatenate
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.utils import Sequence
from keras.applications.vgg16 import VGG16
from keras import preprocessing
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

MODE = 1

np.random.seed(2017)

VOCAB_IMAGE = 3063 + 1
VOCAB_CATEGORY = 47 + 1
VOCAB_PARENT = 9 + 1
EMBED_CATEGORY = 25

def define_model(): 
    # convolution
    image = Input(shape=(150,150,3), name='image')
    imagenet = VGG16(include_top=False, input_tensor=image, pooling='max')
    imagenet = imagenet.output
    # parent
    parent = Input(shape=(1,), name='parent')
    embed_parent = Embedding(VOCAB_PARENT, EMBED_CATEGORY)(parent)
    embed_parent = Flatten()(embed_parent)
    # category
    category = Input(shape=(1,), name='category')
    embed_category = Embedding(VOCAB_CATEGORY, EMBED_CATEGORY)(category)
    embed_category = Flatten()(embed_category)
    # image-top-1
    image_top = Input(shape=(1,), name='image_top')
    embed_image = Embedding(VOCAB_IMAGE, EMBED_CATEGORY)(image_top)
    embed_image = Flatten()(embed_image)
    # concatenate
    concat =  [imagenet, embed_image, embed_category, embed_parent]
    dense = concatenate(concat)
    dense = Dropout(0.3)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(512, activation='relu')(dense)
    dense = Dropout(0.3)(dense)
    dense = Dense(256, activation='relu')(dense)
    dense = Dropout(0.3)(dense)
    predict = Dense(1, activation='sigmoid')(dense)
    # compile
    inputs =  [parent, category, image_top, image]
    model = Model(inputs=inputs, output=predict)
    optimizer = Adam(lr=1e-3, clipnorm=1.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

class DataGenerator(Sequence):
    
    def __init__(self, data, batch_size):
        self.data = data
        self.idx = list(data['image'].values)
        self.batch_size = batch_size
        self.encode = ['parent_category_name','category_name','image_top_1']
        self.on_epoch_end()
        return None

    def __len__(self):
        return int(np.floor(len(self.idx) / self.batch_size))

    def __getitem__(self, index):
        index = self.idx[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(index)
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.idx)
        return None

    def __data_generation(self, index):
        images = pd.DataFrame(index, columns=['image'])
        subset = images.merge(self.data, on='image')
        labels = []
        images = []
        encode_1 = []
        encode_2 = []
        encode_3 = []
        for image in index:
            row = subset.loc[subset['image'] == image].reset_index(drop=True)
            labels.append(row.at[0,'deal_probability'])
            encode_1.append(row.at[0,'parent_category_name'])
            encode_2.append(row.at[0,'category_name'])
            encode_3.append(row.at[0,'image_top_1'])
            path = '../data/download/data/competition_files/train_jpg/' + image + '.jpg'
            image = cv2.imread(path)
            image = preprocess_input(np.array(image, dtype='float64'))
            image = np.array(image, dtype='float16')
            images.append(image)
        image = [np.array(images)]
        labels = np.array(labels)
        encode = [np.array(encode_1), np.array(encode_2), np.array(encode_3)]
        return encode + image, labels

train_data = pd.read_csv('../data/data/image/dataset/train_{}.csv'.format(MODE))
valid_data = pd.read_csv('../data/data/image/dataset/valid_{}.csv'.format(MODE))
train_generator = DataGenerator(train_data, 64)
valid_generator = DataGenerator(valid_data, 64)

model = define_model()
stop = EarlyStopping('val_loss', patience=3, mode="min")
logger = CSVLogger('../data/data/image/model/logger_{}.log'.format(MODE))
best = ModelCheckpoint('../data/data/image/model/weights_{}.h5'.format(MODE), save_best_only=True)
anneal = ReduceLROnPlateau(factor=0.5, patience=1)
callbacks = [stop, logger, best, anneal]

params = {}
params['generator'] = train_generator
params['validation_data'] = valid_generator
params['callbacks'] = callbacks
params['epochs'] = 3
params['use_multiprocessing'] = True
params['workers'] = 8
model.fit_generator(**params)