import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from typing import Tuple, List, Dict, Any
from .config import Config

class XGBoostModel:
    def __init__(self, params: Dict[str, Any] = None, config=None):
        self.config = config or Config()
        self.params = (params if params is not None else self.config.XGB_PARAMS).copy()
        self.params.setdefault('learning_rate', self.config.XGB_LEARNING_RATE)
        self.model = None
        
    def train_cv(self, train_data: pd.DataFrame, target_column: str = 'author') -> pd.DataFrame:
        train_matrix = xgb.DMatrix(
            data=train_data.drop(columns=[target_column, 'id'], errors='ignore'),
            label=train_data[target_column]
        )
        
        cv_params = {
            'params': self.params,
            'dtrain': train_matrix,
            'num_boost_round': self.config.XGB_NUM_ROUNDS * 4,
            'folds': list(StratifiedKFold(n_splits=self.config.N_FOLDS, random_state=self.config.RANDOM_STATE, shuffle=True).split(train_data, train_data[target_column])),
            'early_stopping_rounds': self.config.XGB_EARLY_STOPPING,
            'verbose_eval': 100,
            'show_stdv': False,
        }
        
        cv_results = xgb.cv(**cv_params)
        return cv_results
    
    def train(self, train_data: pd.DataFrame, target_column: str = 'author') -> None:
        train_matrix = xgb.DMatrix(
            data=train_data.drop(columns=[target_column, 'id'], errors='ignore'),
            label=train_data[target_column]
        )
        
        train_params = {
            'params': self.params,
            'dtrain': train_matrix,
            'num_boost_round': self.config.XGB_NUM_ROUNDS,
            'verbose_eval': 200,
        }
        
        self.model = xgb.train(**train_params)
    
    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError('Model must be trained before prediction')
        test_matrix = xgb.DMatrix(data=test_data.drop(columns=['id'], errors='ignore'))
        predictions = self.model.predict(test_matrix)
        
        return pd.DataFrame(predictions, columns=self.config.AUTHOR_NAMES)
    
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        return sorted(self.model.get_fscore().items(), key=lambda x: x[1], reverse=True)

class NaiveBayesModel:
    def __init__(self, config=None):
        self.config = config or Config()
        self.models = {}
        
    def train_cv(self, train_features: np.ndarray, train_targets: np.ndarray,
                 test_features: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_targets = np.asarray(train_targets)
        folds = StratifiedKFold(n_splits=self.config.N_FOLDS, random_state=self.config.RANDOM_STATE, shuffle=True)
        pred_train = np.zeros((len(train_targets), self.config.NUM_CLASSES))
        pred_test = np.zeros((test_features.shape[0], self.config.NUM_CLASSES))
        
        for dev_index, val_index in folds.split(np.zeros(len(train_targets)), train_targets):
            model = MultinomialNB()
            X_train = train_features[dev_index]
            y_train = train_targets[dev_index]
            X_valid = train_features[val_index]
            
            model.fit(X_train, y_train)
            pred_train[val_index, :] = model.predict_proba(X_valid)
            pred_test += model.predict_proba(test_features)
        
        pred_test = pred_test / self.config.N_FOLDS
        
        train_score = pd.DataFrame(pred_train, columns=[f'nb_{i}' for i in range(self.config.NUM_CLASSES)])
        test_score = pd.DataFrame(pred_test, columns=[f'nb_{i}' for i in range(self.config.NUM_CLASSES)])
        
        return train_score, test_score

class NeuralNetworkModel:
    """Original pooled-embedding/LSTM architectures, with aligned OOF outputs."""

    def __init__(self, model_type='simple', config=None):
        if model_type not in {'simple', 'lstm'}:
            raise ValueError('model_type must be simple or lstm')
        self.config = config or Config()
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.embedding_matrix = None

    def _load_glove_embeddings(self, word_index):
        from pathlib import Path

        shape = (len(word_index) + 1, self.config.EMBEDDING_DIM)
        if self.config.RANDOM_EMBEDDINGS:
            matrix = np.random.default_rng(self.config.RANDOM_STATE).normal(0, 0.05, shape).astype('float32')
            matrix[0] = 0
            return matrix
        path = Path(self.config.GLOVE_PATH)
        matrix = np.zeros(shape, dtype='float32')
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                values = line.split()
                if not values or values[0] not in word_index:
                    continue
                if len(values) != self.config.EMBEDDING_DIM + 1:
                    raise ValueError(f'Unexpected GloVe dimension for {values[0]} in {path}')
                matrix[word_index[values[0]]] = np.asarray(values[1:], dtype='float32')
        return matrix

    def _prepare_data(self, train_texts, test_texts, train_targets):
        # TensorFlow stays optional for users running only text/NB/XGBoost steps.
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.utils import pad_sequences, to_categorical

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(list(train_texts) + list(test_texts))
        train_seq = pad_sequences(self.tokenizer.texts_to_sequences(train_texts),
                                  maxlen=self.config.MAX_SEQUENCE_LENGTH)
        test_seq = pad_sequences(self.tokenizer.texts_to_sequences(test_texts),
                                 maxlen=self.config.MAX_SEQUENCE_LENGTH)
        labels = to_categorical(train_targets, num_classes=self.config.NUM_CLASSES)
        return train_seq, labels, test_seq

    def _create_model(self, word_index):
        from tensorflow.keras import Sequential, Input
        from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Embedding, LSTM, Dropout
        from tensorflow.keras.optimizers import Adam

        self.embedding_matrix = self._load_glove_embeddings(word_index)
        model = Sequential([
            Input(shape=(self.config.MAX_SEQUENCE_LENGTH,), dtype='int32'),
            Embedding(len(word_index) + 1, self.config.EMBEDDING_DIM,
                      weights=[self.embedding_matrix], trainable=True),
        ])
        if self.model_type == 'lstm':
            model.add(LSTM(self.config.LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dropout(0.2))
        else:
            model.add(GlobalAveragePooling1D())
        model.add(Dense(self.config.NUM_CLASSES, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=self.config.NN_LEARNING_RATE), metrics=['accuracy'])
        return model

    def train(self, train_texts, test_texts, train_targets):
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        sequences, labels, test_seq = self._prepare_data(train_texts, test_texts, train_targets)
        targets = np.asarray(train_targets)
        folds = StratifiedKFold(n_splits=self.config.N_FOLDS, shuffle=True,
                                random_state=self.config.RANDOM_STATE)
        train_predictions = np.zeros((len(targets), self.config.NUM_CLASSES))
        test_predictions = np.zeros((len(test_texts), self.config.NUM_CLASSES))
        schedule = self.config.NN_SCHEDULE
        if self.config.NN_EPOCHS is not None:
            schedule = [(self.config.NN_LEARNING_RATE, self.config.NN_BATCH_SIZE, self.config.NN_EPOCHS)]
        for fold, (dev, valid) in enumerate(folds.split(sequences, targets)):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(self.config.RANDOM_STATE + fold)
            # Early stopping sees only an inner holdout, never the outer OOF rows.
            inner_train, inner_valid = train_test_split(
                dev, test_size=max(self.config.NUM_CLASSES,
                                   int(np.ceil(len(dev) * self.config.NN_VALIDATION_SPLIT))),
                stratify=targets[dev], random_state=self.config.RANDOM_STATE + fold)
            self.model = self._create_model(self.tokenizer.word_index)
            for lr, batch_size, epochs in schedule:
                self.model.optimizer.learning_rate.assign(lr)
                self.model.fit(sequences[inner_train], labels[inner_train],
                               validation_data=(sequences[inner_valid], labels[inner_valid]),
                               batch_size=batch_size, epochs=epochs, verbose=0,
                               callbacks=[EarlyStopping(patience=2, monitor='val_loss',
                                                        restore_best_weights=True)])
            train_predictions[valid] = self.model.predict(sequences[valid], verbose=0)
            test_predictions += self.model.predict(test_seq, verbose=0) / self.config.N_FOLDS
        columns = [f'{self.model_type}_{i}' for i in range(self.config.NUM_CLASSES)]
        return pd.DataFrame(train_predictions, columns=columns), pd.DataFrame(test_predictions, columns=columns)
