import numpy as np
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.metrics import log_loss

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        
    def create_data_generator(self, transform_params):
        return ImageDataGenerator(**transform_params)
    
    def create_dataflow(self, generator, images, angles, labels):
        flow_1 = generator.flow(images, labels, batch_size=self.config.BATCH_SIZE, seed=self.config.RANDOM_STATE)
        flow_2 = generator.flow(images, angles, batch_size=self.config.BATCH_SIZE, seed=self.config.RANDOM_STATE)
        
        while True:
            tuple_1 = flow_1.next()
            tuple_2 = flow_2.next()
            yield [tuple_1[0], tuple_2[1]], tuple_1[1]
    
    def create_callbacks(self, model_name, fold_idx):
        callbacks = [
            EarlyStopping('val_loss', patience=self.config.PATIENCE, mode="min"),
            ModelCheckpoint(
                f'{self.config.MODEL_DIR}/{model_name}/model_{fold_idx}.hdf5',
                save_best_only=True,
                save_weights_only=True
            ),
            CSVLogger(f'{self.config.MODEL_DIR}/{model_name}/logger_{fold_idx}.log'),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                verbose=0,
                mode='min'
            )
        ]
        return callbacks
    
    def train_model(self, model_class, model_name, source_name, transform_params):
        os.makedirs(f'{self.config.MODEL_DIR}/{model_name}', exist_ok=True)
        
        for fold_idx in range(1, self.config.FOLDS + 1):
            train_images = np.load(f'{self.config.DATA_DIR}/{source_name}/train/train_images_{fold_idx}.npy')
            train_angles = np.load(f'{self.config.DATA_DIR}/{source_name}/train/train_angles_{fold_idx}.npy')
            train_labels = np.load(f'{self.config.DATA_DIR}/{source_name}/train/train_labels_{fold_idx}.npy')
            
            test_images = np.load(f'{self.config.DATA_DIR}/{source_name}/train/test_images_{fold_idx}.npy')
            test_angles = np.load(f'{self.config.DATA_DIR}/{source_name}/train/test_angles_{fold_idx}.npy')
            test_labels = np.load(f'{self.config.DATA_DIR}/{source_name}/train/test_labels_{fold_idx}.npy')
            
            generator = self.create_data_generator(transform_params)
            train_generator = self.create_dataflow(generator, train_images, train_angles, train_labels)
            test_generator = ([test_images, test_angles], test_labels)
            
            model = model_class.define_model()
            callbacks = self.create_callbacks(model_name, fold_idx)
            
            model.fit_generator(
                generator=train_generator,
                validation_data=test_generator,
                steps_per_epoch=self.config.MODEL_CONFIGS[model_name]['steps_per_epoch'],
                epochs=self.config.MODEL_CONFIGS[model_name]['epochs'],
                verbose=0,
                callbacks=callbacks
            )
            
            K.clear_session()
    
    def train_vgg16_model(self, model_class, model_name, source_name, transform_params):
        os.makedirs(f'{self.config.MODEL_DIR}/{model_name}', exist_ok=True)
        
        for fold_idx in range(1, self.config.FOLDS + 1):
            train_images = np.load(f'{self.config.DATA_DIR}/{source_name}/train/train_images_{fold_idx}.npy')
            train_angles = np.load(f'{self.config.DATA_DIR}/{source_name}/train/train_angles_{fold_idx}.npy')
            train_labels = np.load(f'{self.config.DATA_DIR}/{source_name}/train/train_labels_{fold_idx}.npy')
            
            test_images = np.load(f'{self.config.DATA_DIR}/{source_name}/train/test_images_{fold_idx}.npy')
            test_angles = np.load(f'{self.config.DATA_DIR}/{source_name}/train/test_angles_{fold_idx}.npy')
            test_labels = np.load(f'{self.config.DATA_DIR}/{source_name}/train/test_labels_{fold_idx}.npy')
            
            generator = self.create_data_generator(transform_params)
            train_generator = self.create_dataflow(generator, train_images, train_angles, train_labels)
            test_generator = ([test_images, test_angles], test_labels)
            
            callbacks = self.create_callbacks(model_name, fold_idx)
            
            model = model_class.define_model(trainable=False, learning_rate=1e-4)
            model.fit_generator(
                generator=train_generator,
                validation_data=test_generator,
                steps_per_epoch=self.config.MODEL_CONFIGS[model_name]['steps_per_epoch'],
                epochs=self.config.MODEL_CONFIGS[model_name]['epochs'],
                verbose=0,
                callbacks=callbacks
            )
            
            model = model_class.define_model(trainable=True, learning_rate=5e-5)
            model.load_weights(f'{self.config.MODEL_DIR}/{model_name}/model_{fold_idx}.hdf5')
            model.fit_generator(
                generator=train_generator,
                validation_data=test_generator,
                steps_per_epoch=self.config.MODEL_CONFIGS[model_name]['steps_per_epoch'],
                epochs=self.config.MODEL_CONFIGS[model_name]['fine_tune_epochs'],
                verbose=0,
                callbacks=callbacks
            )
            
            K.clear_session()
