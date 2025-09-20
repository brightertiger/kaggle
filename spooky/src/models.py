import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import log_loss
import keras
from keras.layers import Dense, GlobalAveragePooling1D, Embedding, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import Adam
import os
from typing import Tuple, List, Dict, Any
from .config import Config

class XGBoostModel:
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or Config.XGB_PARAMS.copy()
        self.model = None
        
    def train_cv(self, train_data: pd.DataFrame, target_column: str = 'author') -> Dict[str, Any]:
        train_matrix = xgb.DMatrix(
            data=train_data.drop(columns=[target_column]), 
            label=train_data[target_column]
        )
        
        cv_params = {
            'params': self.params,
            'dtrain': train_matrix,
            'num_boost_round': Config.XGB_NUM_ROUNDS * 4,
            'folds': KFold(n_splits=Config.N_FOLDS, random_state=Config.RANDOM_STATE, shuffle=True).split(train_data),
            'early_stopping_rounds': Config.XGB_EARLY_STOPPING,
            'verbose_eval': 100,
            'show_stdv': False,
            'callbacks': [xgb.callback.reset_learning_rate([Config.XGB_LEARNING_RATE] * Config.XGB_NUM_ROUNDS * 4)]
        }
        
        cv_results = xgb.cv(**cv_params)
        return cv_results
    
    def train(self, train_data: pd.DataFrame, target_column: str = 'author') -> None:
        train_matrix = xgb.DMatrix(
            data=train_data.drop(columns=[target_column]), 
            label=train_data[target_column]
        )
        
        train_params = {
            'params': self.params,
            'dtrain': train_matrix,
            'num_boost_round': Config.XGB_NUM_ROUNDS,
            'verbose_eval': 200,
            'callbacks': [xgb.callback.reset_learning_rate([Config.XGB_LEARNING_RATE] * Config.XGB_NUM_ROUNDS)]
        }
        
        self.model = xgb.train(**train_params)
    
    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        test_matrix = xgb.DMatrix(data=test_data)
        predictions = self.model.predict(test_matrix)
        
        return pd.DataFrame(predictions, columns=Config.AUTHOR_NAMES)
    
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        return sorted(self.model.get_fscore().items(), key=lambda x: x[1], reverse=True)

class NaiveBayesModel:
    def __init__(self):
        self.models = {}
        
    def train_cv(self, train_features: np.ndarray, train_targets: np.ndarray, 
                 test_features: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        folds = KFold(n_splits=Config.N_FOLDS, random_state=Config.RANDOM_STATE, shuffle=True)
        pred_train = np.zeros((len(train_targets), Config.NUM_CLASSES))
        pred_test = np.zeros((test_features.shape[0], Config.NUM_CLASSES))
        
        for dev_index, val_index in folds.split(train_features):
            model = MultinomialNB()
            X_train = train_features[dev_index]
            y_train = train_targets[dev_index]
            X_valid = train_features[val_index]
            
            model.fit(X_train, y_train)
            pred_train[val_index, :] = model.predict_proba(X_valid)
            pred_test += model.predict_proba(test_features)
        
        pred_test = pred_test / Config.N_FOLDS
        
        train_score = pd.DataFrame(pred_train, columns=[f'nb_{i}' for i in range(Config.NUM_CLASSES)])
        test_score = pd.DataFrame(pred_test, columns=[f'nb_{i}' for i in range(Config.NUM_CLASSES)])
        
        return train_score, test_score

class NeuralNetworkModel:
    def __init__(self, model_type: str = 'simple'):
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.embedding_matrix = None
        
    def _load_glove_embeddings(self, word_index: Dict[str, int]) -> np.ndarray:
        embeddings_index = {}
        
        glove_path = Config.GLOVE_PATH
        if not glove_path.exists():
            raise FileNotFoundError(f"Glove embeddings not found at {glove_path}")
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        embedding_matrix = np.zeros((len(word_index) + 1, Config.EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        return embedding_matrix
    
    def _prepare_data(self, train_texts: List[str], test_texts: List[str], 
                     train_targets: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(train_texts + test_texts)
        
        train_sequences = self.tokenizer.texts_to_sequences(train_texts)
        test_sequences = self.tokenizer.texts_to_sequences(test_texts)
        
        train_seq = pad_sequences(train_sequences, maxlen=Config.MAX_SEQUENCE_LENGTH)
        test_seq = pad_sequences(test_sequences, maxlen=Config.MAX_SEQUENCE_LENGTH)
        
        labels = to_categorical(train_targets, num_classes=Config.NUM_CLASSES)
        
        indices = np.arange(train_seq.shape[0])
        np.random.shuffle(indices)
        train_seq = train_seq[indices]
        labels = labels[indices]
        
        nb_validation_samples = int(Config.NN_VALIDATION_SPLIT * train_seq.shape[0])
        x_train = train_seq[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]
        x_val = train_seq[-nb_validation_samples:]
        y_val = labels[-nb_validation_samples:]
        
        return x_train, y_train, x_val, y_val, test_seq
    
    def _create_model(self, word_index: Dict[str, int]) -> Sequential:
        self.embedding_matrix = self._load_glove_embeddings(word_index)
        
        embedding_layer = Embedding(
            len(word_index) + 1, 
            Config.EMBEDDING_DIM, 
            weights=[self.embedding_matrix], 
            input_length=Config.MAX_SEQUENCE_LENGTH, 
            trainable=True
        )
        
        model = Sequential()
        model.add(embedding_layer)
        
        if self.model_type == 'lstm':
            model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dropout(0.2))
        else:
            model.add(GlobalAveragePooling1D())
        
        model.add(Dense(Config.NUM_CLASSES, activation='softmax'))
        
        optimizer = Adam(lr=Config.NN_LEARNING_RATE)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_texts: List[str], test_texts: List[str], 
              train_targets: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x_train, y_train, x_val, y_val, test_seq = self._prepare_data(
            train_texts, test_texts, train_targets
        )
        
        self.model = self._create_model(self.tokenizer.word_index)
        
        learning_rates = [0.0001, 0.001, 0.0005, 0.0003, 0.0002]
        batch_sizes = [8, 8, 16, 16, 32]
        epochs_list = [3, 3, 10, 10, 5]
        
        for lr, batch_size, epochs in zip(learning_rates, batch_sizes, epochs_list):
            self.model.optimizer.lr = lr
            
            params = {
                'validation_data': (x_val, y_val),
                'batch_size': batch_size,
                'epochs': epochs,
                'callbacks': [EarlyStopping(patience=2, monitor='val_loss')],
                'verbose': 0
            }
            
            self.model.fit(x_train, y_train, **params)
        
        train_predictions = self.model.predict(x_train)
        test_predictions = self.model.predict(test_seq)
        
        train_df = pd.DataFrame(train_predictions, columns=[f'{self.model_type}_{i}' for i in range(Config.NUM_CLASSES)])
        test_df = pd.DataFrame(test_predictions, columns=[f'{self.model_type}_{i}' for i in range(Config.NUM_CLASSES)])
        
        return train_df, test_df
