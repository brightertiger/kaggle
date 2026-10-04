import numpy as np
from keras.models import Model
from keras.layers import (Dense, Dropout, Flatten, Activation, Conv2D, 
                         MaxPooling2D, Input, concatenate, BatchNormalization)
from keras.optimizers import Adam
from keras.applications import VGG16
import xgboost as xgb

class CNNBasic:
    def __init__(self, config):
        self.config = config
        
    def define_model(self, compile_model=True):
        input_1 = Input(shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3), name='image')
        input_2 = Input(shape=(1,), name='angle')
        angle = Dense(1)(input_2)
        
        convolve = Conv2D(max(1, int(64 * self.config.CNN_WIDTH)), kernel_size=(3, 3), padding='same')(input_1)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
        
        convolve = Conv2D(max(1, int(128 * self.config.CNN_WIDTH)), kernel_size=(3, 3), padding='same')(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
        
        convolve = Conv2D(max(1, int(256 * self.config.CNN_WIDTH)), kernel_size=(3, 3), padding='same')(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = Conv2D(max(1, int(256 * self.config.CNN_WIDTH)), kernel_size=(3, 3), padding='same')(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
        
        convolve = Conv2D(max(1, int(512 * self.config.CNN_WIDTH)), kernel_size=(3, 3), padding='same')(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = Conv2D(max(1, int(512 * self.config.CNN_WIDTH)), kernel_size=(3, 3), padding='same')(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
        
        convolve = Flatten()(convolve)
        convolve = Dropout(0.3)(convolve)
        
        concat = concatenate([convolve, angle])
        concat = Dense(max(1, int(512 * self.config.CNN_WIDTH)), activation='swish', kernel_initializer='he_normal')(concat)
        concat = Dropout(0.3)(concat)
        concat = Dense(max(1, int(256 * self.config.CNN_WIDTH)), activation='swish', kernel_initializer='he_normal')(concat)
        concat = Dropout(0.3)(concat)
        predict = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(concat)
        
        model = Model(inputs=[input_1, input_2], outputs=predict)
        if compile_model:
            optimizer = Adam(learning_rate=self.config.LEARNING_RATE)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

class CNNAdvanced:
    def __init__(self, config):
        self.config = config
        
    def define_model(self, compile_model=True):
        input_1 = Input(shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3), name='image')
        input_2 = Input(shape=(1,), name='angle')
        angle = Dense(1)(input_2)
        
        convolve = Conv2D(max(1, int(64 * self.config.CNN_WIDTH)), kernel_size=(3, 3))(input_1)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = Conv2D(max(1, int(64 * self.config.CNN_WIDTH)), kernel_size=(3, 3))(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
        
        convolve = Conv2D(max(1, int(256 * self.config.CNN_WIDTH)), kernel_size=(3, 3))(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = Conv2D(max(1, int(256 * self.config.CNN_WIDTH)), kernel_size=(3, 3))(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
        
        convolve = Conv2D(max(1, int(512 * self.config.CNN_WIDTH)), kernel_size=(3, 3))(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = Conv2D(max(1, int(512 * self.config.CNN_WIDTH)), kernel_size=(3, 3))(convolve)
        convolve = BatchNormalization()(convolve)
        convolve = Activation('swish')(convolve)
        convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
        
        convolve = Flatten()(convolve)
        convolve = Dropout(0.3)(convolve)
        
        concat = concatenate([convolve, angle])
        concat = Dense(max(1, int(256 * self.config.CNN_WIDTH)), activation='swish', kernel_initializer='he_normal')(concat)
        concat = Dropout(0.3)(concat)
        concat = Dense(max(1, int(128 * self.config.CNN_WIDTH)), activation='swish', kernel_initializer='he_normal')(concat)
        concat = Dropout(0.3)(concat)
        predict = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(concat)
        
        model = Model(inputs=[input_1, input_2], outputs=predict)
        if compile_model:
            optimizer = Adam(learning_rate=self.config.LEARNING_RATE)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

class VGG16Model:
    def __init__(self, config):
        self.config = config
        
    def define_model(self, trainable=False, learning_rate=1e-4, pretrained=True, compile_model=True):
        input_1 = Input(shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3), name='image')
        input_2 = Input(shape=(1,), name='angle')
        angle = Dense(1)(input_2)
        
        if self.config.VGG_WIDTH != 1.0:
            if self.config.VGG_WEIGHTS is not None:
                raise ValueError('Reduced VGG width requires VGG_WEIGHTS=None')
            convolve = input_1
            for block, (filters, depth) in enumerate(zip((64, 128, 256, 512, 512), (2, 2, 3, 3, 3))):
                for layer in range(depth):
                    convolve = Conv2D(max(1, int(filters * self.config.VGG_WIDTH)), 3,
                                      activation='relu', padding='same',
                                      name=f'block{block + 1}_conv{layer + 1}')(convolve)
                convolve = MaxPooling2D(2)(convolve)
            from keras.layers import GlobalMaxPooling2D
            vgg = Model(input_1, GlobalMaxPooling2D()(convolve))
        else:
            vgg = VGG16(input_tensor=input_1, pooling='max', include_top=False,
                        weights=self.config.VGG_WEIGHTS if pretrained else None)
        for layer in vgg.layers:
            layer.trainable = trainable
            
        convolve = Dropout(0.3)(vgg.output)
        
        concat = concatenate([convolve, angle])
        concat = Dense(max(1, int(512 * self.config.CNN_WIDTH)), activation='swish', kernel_initializer='he_normal')(concat)
        concat = Dropout(0.2)(concat)
        concat = Dense(max(1, int(256 * self.config.CNN_WIDTH)), activation='swish', kernel_initializer='he_normal')(concat)
        concat = Dropout(0.2)(concat)
        predict = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(concat)
        
        model = Model(inputs=[input_1, input_2], outputs=predict)
        if compile_model:
            optimizer = Adam(learning_rate=learning_rate)
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
    
    def xgboost_stack(self, train_data, test_data):
        # Labels and IDs must never enter the feature matrix.
        columns = [c for c in train_data.columns if c not in ('id', 'label')]
        if set(columns) != set(test_data.columns) - {'id'}:
            raise ValueError('Train/test stacking features do not match')
        train_matrix = xgb.DMatrix(train_data[columns], label=train_data['label'])
        test_matrix = xgb.DMatrix(test_data[columns])
        cv_results = xgb.cv(
            params=self.config.XGBOOST_PARAMS,
            dtrain=train_matrix,
            num_boost_round=self.config.XGB_ROUNDS,
            early_stopping_rounds=self.config.XGB_EARLY_STOPPING,
            verbose_eval=False,
            nfold=self.config.FOLDS,
            stratified=True,
            seed=self.config.RANDOM_STATE,
        )
        model = xgb.train(self.config.XGBOOST_PARAMS, train_matrix,
                          num_boost_round=len(cv_results))
        from pathlib import Path
        Path(self.config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        model.save_model(str(Path(self.config.MODEL_DIR) / 'xgboost.json'))
        return model.predict(test_matrix)
