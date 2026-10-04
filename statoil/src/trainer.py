"""Keras training with synchronized image/angle augmentation and fold checkpoints."""
import os

import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam
from keras.utils import set_random_seed
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def create_data_generator(self, transform_params):
        return ImageDataGenerator(**transform_params)

    def create_dataflow(self, generator, images, angles, labels):
        # One shuffled iterator keeps image, angle, and label paired exactly.
        targets = np.column_stack((angles, labels)).astype(np.float32)
        flow = generator.flow(images, targets, batch_size=self.config.BATCH_SIZE,
                              seed=self.config.RANDOM_STATE)
        while True:
            batch_images, batch_targets = next(flow)
            yield (batch_images, batch_targets[:, :1]), batch_targets[:, 1:]

    def create_callbacks(self, model_name, fold_idx):
        folder = f'{self.config.MODEL_DIR}/{model_name}'
        return [
            EarlyStopping('val_loss', patience=self.config.MODEL_CONFIGS[model_name]['patience'],
                          mode='min'),
            ModelCheckpoint(f'{folder}/model_{fold_idx}.weights.h5',
                            save_best_only=True, save_weights_only=True),
            CSVLogger(f'{folder}/logger_{fold_idx}.log', append=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, mode='min'),
        ]

    def _train(self, model_class, model_name, source_name, transform_params, fine_tune):
        os.makedirs(f'{self.config.MODEL_DIR}/{model_name}', exist_ok=True)
        settings = self.config.MODEL_CONFIGS[model_name]
        for fold_idx in range(1, self.config.FOLDS + 1):
            set_random_seed(self.config.RANDOM_STATE + fold_idx)
            folder = f'{self.config.DATA_DIR}/{source_name}/train'
            def load(prefix, name):
                return np.load(f'{folder}/{prefix}_{name}_{fold_idx}.npy')
            generator = self.create_data_generator(transform_params)
            train_generator = self.create_dataflow(
                generator, load('train', 'images'), load('train', 'angles'), load('train', 'labels'))
            validation = ((load('test', 'images'), load('test', 'angles').reshape(-1, 1)),
                          load('test', 'labels').reshape(-1, 1))
            model = model_class.define_model()
            callbacks = self.create_callbacks(model_name, fold_idx)
            print(f'Training {model_name}, fold {fold_idx}', flush=True)
            model.fit(train_generator, validation_data=validation,
                      steps_per_epoch=settings['steps_per_epoch'], epochs=settings['epochs'],
                      verbose=0, callbacks=callbacks, shuffle=False)
            if fine_tune and settings['fine_tune_epochs'] > 0:
                # Keep the same checkpoint callback so fine-tuning cannot overwrite a
                # better frozen-stage checkpoint. Recompile after unfreezing.
                model.load_weights(f'{self.config.MODEL_DIR}/{model_name}/model_{fold_idx}.weights.h5')
                for layer in model.layers:
                    layer.trainable = True
                model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=5e-5),
                              metrics=['accuracy'])
                model.fit(train_generator, validation_data=validation,
                          steps_per_epoch=settings['steps_per_epoch'],
                          epochs=settings['fine_tune_epochs'], verbose=0, callbacks=callbacks, shuffle=False)
            K.clear_session()

    def train_model(self, model_class, model_name, source_name, transform_params):
        self._train(model_class, model_name, source_name, transform_params, fine_tune=False)

    def train_vgg16_model(self, model_class, model_name, source_name, transform_params):
        self._train(model_class, model_name, source_name, transform_params, fine_tune=True)
