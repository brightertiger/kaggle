import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import (Dense, Dropout, Flatten, Activation, Conv2D, 
                         MaxPooling2D, Input, concatenate, BatchNormalization)
from keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects
from keras.applications import VGG16
from keras import backend as K
from sklearn.metrics import log_loss
import xgboost as xgb

def swish(x):
    return K.sigmoid(x) * x

get_custom_objects().update({'swish': Activation(swish)})

class CNNBasic:
    def __init__(self, config):
        self.config = config
        
    def define_model(self):
        input_1 = Input(shape=(75, 75, 3), name='image')
        input_2 = Input(shape=(1,), name='angle')
        angle = Dense(1)(input_2)
        
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
        
        concat = concatenate([convolve, angle])
        concat = Dense(512, activation='swish', kernel_initializer='he_normal')(concat)
        concat = Dropout(0.3)(concat)
        concat = Dense(256, activation='swish', kernel_initializer='he_normal')(concat)
        concat = Dropout(0.3)(concat)
        predict = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(concat)
        
        model = Model(inputs=[input_1, input_2], output=predict)
        optimizer = Adam(lr=self.config.LEARNING_RATE)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

class CNNAdvanced:
    def __init__(self, config):
        self.config = config
        
    def define_model(self):
        input_1 = Input(shape=(75, 75, 3), name='image')
        input_2 = Input(shape=(1,), name='angle')
        angle = Dense(1)(input_2)
        
        convolve = Conv2D(64, kernel_size=(3, 3))(input_1)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = Conv2D(64, kernel_size=(3, 3))(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
        
        convolve = Conv2D(256, kernel_size=(3, 3))(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = Conv2D(256, kernel_size=(3, 3))(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
        
        convolve = Conv2D(512, kernel_size=(3, 3))(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = Conv2D(512, kernel_size=(3, 3))(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
        
        convolve = Flatten()(convolve)
        convolve = Dropout(0.3)(convolve)
        
        concat = concatenate([convolve, angle])
        concat = Dense(256, activation='swish', kernel_initializer='he_normal')(concat)
        concat = Dropout(0.3)(concat)
        concat = Dense(128, activation='swish', kernel_initializer='he_normal')(concat)
        concat = Dropout(0.3)(concat)
        predict = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(concat)
        
        model = Model(inputs=[input_1, input_2], output=predict)
        optimizer = Adam(lr=self.config.LEARNING_RATE)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

class VGG16Model:
    def __init__(self, config):
        self.config = config
        
    def define_model(self, trainable=False, learning_rate=1e-4):
        input_1 = Input(shape=(75, 75, 3), name='image')
        input_2 = Input(shape=(1,), name='angle')
        angle = Dense(1)(input_2)
        
        vgg = VGG16(input_tensor=input_1, pooling='max', include_top=False)
        for layer in vgg.layers:
            layer.trainable = trainable
            
        convolve = Dropout(0.3)(vgg.output)
        
        concat = concatenate([convolve, angle])
        concat = Dense(512, activation='swish', kernel_initializer='he_normal')(concat)
        concat = Dropout(0.2)(concat)
        concat = Dense(256, activation='swish', kernel_initializer='he_normal')(concat)
        concat = Dropout(0.2)(concat)
        predict = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(concat)
        
        model = Model(inputs=[input_1, input_2], output=predict)
        optimizer = Adam(lr=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

class EnsembleModel:
    def __init__(self, config):
        self.config = config
        
    def simple_stack(self, train_scores, test_scores, low_threshold=0.15, high_threshold=0.95):
        def stack_func(values, low, high):
            if np.all(values < low):
                return np.min(values)
            elif np.all(values > high):
                return np.max(values)
            else:
                return np.mean(values)
        
        train_stacked = train_scores.apply(
            lambda x: stack_func(x, low_threshold, high_threshold), axis=1
        ).clip(0.001, 0.999)
        
        test_stacked = test_scores.apply(
            lambda x: stack_func(x, low_threshold, high_threshold), axis=1
        ).clip(0.001, 0.999)
        
        return train_stacked, test_stacked
    
    def xgboost_stack(self, train_data, test_data, model_scores):
        train_matrix = xgb.DMatrix(data=train_data.iloc[:, 2:], label=train_data['label'])
        test_matrix = xgb.DMatrix(data=test_data.iloc[:, 1:])
        
        cv_results = xgb.cv(
            params=self.config.XGBOOST_PARAMS,
            dtrain=train_matrix,
            num_boost_round=3000,
            early_stopping_rounds=200,
            verbose_eval=400,
            nfold=5
        )
        
        best_rounds = cv_results.shape[0]
        
        model = xgb.train(
            params=self.config.XGBOOST_PARAMS,
            dtrain=train_matrix,
            num_boost_round=best_rounds
        )
        
        predictions = model.predict(test_matrix)
        return predictions
