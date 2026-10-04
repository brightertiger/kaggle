import pandas as pd
import numpy as np
import re
import string
from typing import Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .config import Config
from .ensemble import ModelEvaluator, align_predictions


class NeuralNetworkModel:
    """Neural network model for toxic comment classification"""

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None

        # Import lazily so sparse-only training does not initialize TensorFlow.
        import tensorflow as tf
        if config.model.cpu_only:
            tf.config.set_visible_devices([], 'GPU')

    def load_embeddings(self, embedding_path: str) -> Dict[str, np.ndarray]:
        """Load word embeddings from file"""
        embeddings_index = {}

        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                if not values:
                    continue
                # FastText text files may start with a vocabulary/dimension header.
                if len(values) == 2 and all(value.isdigit() for value in values):
                    continue
                if len(values) != self.config.model.embed_size + 1:
                    raise ValueError(f'Embedding dimension mismatch in {embedding_path}')
                word = values[0]
                embeddings_index[word] = np.asarray(values[1:], dtype='float32')
        if not embeddings_index:
            raise ValueError(f'No embeddings found in {embedding_path}')

        return embeddings_index

    def create_embedding_matrix(self, tokenizer,
                              embeddings_index: Dict[str, np.ndarray]) -> np.ndarray:
        """Create embedding matrix from tokenizer and embeddings"""
        word_index = tokenizer.word_index

        # Initialize with random normal distribution
        if embeddings_index:
            # Running moments avoid copying the entire pretrained vector table.
            count = sum(vector.size for vector in embeddings_index.values())
            mean = sum(vector.sum(dtype=np.float64) for vector in embeddings_index.values()) / count
            second = sum(np.square(vector.astype(np.float64)).sum()
                         for vector in embeddings_index.values()) / count
            std = np.sqrt(max(second - mean ** 2, 0.0))
        elif self.config.model.random_embeddings:
            mean, std = 0.0, 0.05
        else:
            raise ValueError('Pretrained embeddings are required unless random_embeddings is enabled')

        size = min(len(word_index) + 1, self.config.model.usable_vocab)
        embedding_matrix = np.random.normal(mean, std, (size, self.config.model.embed_size)).astype('float32')

        # Fill in known embeddings
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None and i < size:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def prepare_data(self, train_text: pd.DataFrame, valid_text: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Prepare text data for neural network"""
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        train_text_list = train_text['comment_text'].fillna('nan').astype(str).tolist()
        valid_text_list = valid_text['comment_text'].fillna('nan').astype(str).tolist()

        self.tokenizer = Tokenizer(lower=True, char_level=False,
                                      num_words=self.config.model.usable_vocab)
        self.tokenizer.fit_on_texts(train_text_list + valid_text_list)

        train_tokens = self.tokenizer.texts_to_sequences(train_text_list)
        valid_tokens = self.tokenizer.texts_to_sequences(valid_text_list)

        train_seq = pad_sequences(train_tokens, maxlen=self.config.model.seq_length)
        valid_seq = pad_sequences(valid_tokens, maxlen=self.config.model.seq_length)

        return train_seq, valid_seq, self.tokenizer.word_index

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the fitted vocabulary and the same padding at prediction time."""
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        tokens = self.tokenizer.texts_to_sequences(frame['comment_text'].fillna('nan').astype(str))
        return pad_sequences(tokens, maxlen=self.config.model.seq_length)

    def build_model(self, embedding_matrix: np.ndarray):
        """Build neural network architecture"""
        from tensorflow.keras.layers import (
            Dense, Dropout, GRU, Embedding, Input, concatenate,
            GlobalAveragePooling1D, Bidirectional, GlobalMaxPooling1D,
            SpatialDropout1D, BatchNormalization,
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        rnn_params = {
            'units': self.config.model.rnn_units,
            'return_sequences': True,
            'recurrent_dropout': self.config.model.recurrent_dropout,
            'dropout': self.config.model.dropout,
            'activation': 'tanh'
        }

        inputs = Input(shape=(self.config.model.seq_length,), dtype='int32', name='sequence')
        embed = Embedding(len(embedding_matrix), self.config.model.embed_size,
                         weights=[embedding_matrix], trainable=False)(inputs)
        embed = SpatialDropout1D(0.2)(embed)

        # First GRU layer with return_sequences=True
        lstm_1 = Bidirectional(GRU(**rnn_params))(embed)
        lstm_1 = BatchNormalization()(lstm_1)

        # Parallel second GRU branch, also reading the embeddings.
        rnn_params['return_sequences'] = False
        lstm_2 = Bidirectional(GRU(**rnn_params))(embed)
        lstm_2 = BatchNormalization()(lstm_2)

        # Pooling layers
        max_pool = GlobalMaxPooling1D()(lstm_1)
        avg_pool = GlobalAveragePooling1D()(lstm_1)

        # Concatenate features
        pool = concatenate([lstm_2, max_pool, avg_pool])
        pool = BatchNormalization()(pool)
        lstm = Dropout(0.2)(pool)

        # Dense layers
        dense = Dense(self.config.model.dense_units, activation='swish')(lstm)
        dense = Dropout(0.2)(dense)
        dense = Dense(self.config.model.dense_units, activation='swish')(dense)
        dense = Dropout(0.2)(dense)

        # Output layer
        predict = Dense(len(self.config.evaluation.target_columns), activation='sigmoid')(dense)

        model = Model(inputs=inputs, outputs=predict)
        optimizer = Adam(learning_rate=self.config.model.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

        return model


class NaiveBayesSVM:
    """Naive Bayes SVM model for toxic comment classification"""

    def __init__(self, config: Config):
        self.config = config

    def tokenize(self, text: str) -> list:
        """Custom tokenizer for NB-SVM"""
        re_tok = re.compile('([' + re.escape(string.punctuation + '¨«»®´·º½¾¿¡§£₤') + '])')
        return re_tok.sub(r' \1 ', text).split()

    def probability(self, document, y_i: int, y: np.ndarray) -> np.ndarray:
        """Calculate word probabilities for class y_i"""
        p = document[y == y_i].sum(0)
        return (p + 1) / ((y == y_i).sum() + 1)

    def compute_features(self, document, labels: pd.Series) -> Tuple[LogisticRegression, np.ndarray]:
        """Compute NB-SVM features"""
        y = labels.values
        r = np.asarray(np.log(self.probability(document, 1, y) / self.probability(document, 0, y))).ravel()
        m = LogisticRegression(C=4, dual=True, solver='liblinear',
                               random_state=self.config.data.random_state, max_iter=1000)
        x_nb = document.multiply(r)
        return m.fit(x_nb, y), r

    def train_model(self, train_labels: pd.DataFrame, train_text: pd.DataFrame,
                   valid_text: pd.DataFrame, test_text: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train NB-SVM model"""
        params = {
            'ngram_range': (1, 2),
            'tokenizer': self.tokenize,
            'token_pattern': None,
            'min_df': self.config.model.nb_min_df,
            'max_df': self.config.model.nb_max_df,
            'strip_accents': 'unicode',
            'use_idf': True,
            'smooth_idf': True,
            'sublinear_tf': True
        }

        vectorizer = TfidfVectorizer(**params)
        train_doc = vectorizer.fit_transform(train_text['comment_text'].fillna('nan'))
        valid_doc = vectorizer.transform(valid_text['comment_text'].fillna('nan'))
        test_doc = vectorizer.transform(test_text['comment_text'].fillna('nan'))

        targets = self.config.evaluation.target_columns
        train_labels = align_predictions(train_labels, train_text['id'], targets)
        valid_ids = valid_text[['id']].reset_index(drop=True)
        test_ids = test_text[['id']].reset_index(drop=True)

        valid_score = np.zeros([valid_text.shape[0], len(targets)])
        test_score = np.zeros([test_text.shape[0], len(targets)])

        for idx, col in enumerate(targets):
            if train_labels[col].nunique() == 1:
                valid_score[:, idx] = test_score[:, idx] = train_labels[col].iloc[0]
                continue
            model, result = self.compute_features(train_doc, train_labels[col])
            valid_score[:, idx] = model.predict_proba(valid_doc.multiply(result))[:, 1]
            test_score[:, idx] = model.predict_proba(test_doc.multiply(result))[:, 1]

        valid_score_df = pd.DataFrame(valid_score)
        valid_score_df.columns = targets
        valid_score_df = valid_ids.join(valid_score_df)

        test_score_df = pd.DataFrame(test_score)
        test_score_df.columns = targets
        test_score_df = test_ids.join(test_score_df)

        return valid_score_df, test_score_df


class LogisticRegressionModel:
    """Logistic Regression with TF-IDF features"""

    def __init__(self, config: Config):
        self.config = config

    def train_model(self, train_labels: pd.DataFrame, train_text: pd.DataFrame,
                   valid_text: pd.DataFrame, test_text: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train logistic regression model with word and char features"""

        # Word-level TF-IDF
        word_params = {
            'analyzer': 'word',
            'ngram_range': (1, 1),
            'token_pattern': r'\w{1,}',
            'stop_words': 'english',
            'strip_accents': 'unicode',
            'sublinear_tf': True,
            'max_features': self.config.model.word_max_features
        }
        word_vectorizer = TfidfVectorizer(**word_params)

        # Char-level TF-IDF
        char_params = {
            'analyzer': 'char',
            'ngram_range': (2, 6),
            'strip_accents': 'unicode',
            'sublinear_tf': True,
            'max_features': self.config.model.char_max_features
        }
        char_vectorizer = TfidfVectorizer(**char_params)

        # Fit vectorizers
        all_text = pd.concat([
            train_text['comment_text'].fillna('nan'),
            valid_text['comment_text'].fillna('nan'),
            test_text['comment_text'].fillna('nan')
        ])

        word_vectorizer.fit(all_text)
        char_vectorizer.fit(all_text)

        # Transform data
        train_word_doc = word_vectorizer.transform(train_text['comment_text'].fillna('nan'))
        valid_word_doc = word_vectorizer.transform(valid_text['comment_text'].fillna('nan'))
        test_word_doc = word_vectorizer.transform(test_text['comment_text'].fillna('nan'))

        train_char_doc = char_vectorizer.transform(train_text['comment_text'].fillna('nan'))
        valid_char_doc = char_vectorizer.transform(valid_text['comment_text'].fillna('nan'))
        test_char_doc = char_vectorizer.transform(test_text['comment_text'].fillna('nan'))

        # Combine features
        from scipy.sparse import hstack
        train_features = hstack([train_char_doc, train_word_doc])
        valid_features = hstack([valid_char_doc, valid_word_doc])
        test_features = hstack([test_char_doc, test_word_doc])

        targets = self.config.evaluation.target_columns
        train_labels = align_predictions(train_labels, train_text['id'], targets)
        valid_ids = valid_text[['id']].reset_index(drop=True)
        test_ids = test_text[['id']].reset_index(drop=True)

        valid_score = np.zeros([valid_text.shape[0], len(targets)])
        test_score = np.zeros([test_text.shape[0], len(targets)])

        for idx, col in enumerate(targets):
            if train_labels[col].nunique() == 1:
                valid_score[:, idx] = test_score[:, idx] = train_labels[col].iloc[0]
                continue
            model = LogisticRegression(C=0.1, solver='sag', max_iter=1000,
                                       random_state=self.config.data.random_state)
            model.fit(train_features, train_labels[col].values)
            valid_score[:, idx] = model.predict_proba(valid_features)[:, 1]
            test_score[:, idx] = model.predict_proba(test_features)[:, 1]

        valid_score_df = pd.DataFrame(valid_score)
        valid_score_df.columns = targets
        valid_score_df = valid_ids.join(valid_score_df)

        test_score_df = pd.DataFrame(test_score)
        test_score_df.columns = targets
        test_score_df = test_ids.join(test_score_df)

        return valid_score_df, test_score_df


def evaluate_predictions(predictions: pd.DataFrame, actual: pd.DataFrame,
                        target_columns: list) -> Dict[str, float]:
    """Evaluate model predictions using ROC AUC"""
    return ModelEvaluator(Config()).evaluate_single_model(predictions, actual, target_columns)
