import pandas as pd
import numpy as np
import re
import string
from typing import Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from keras.layers import (Dense, Dropout, GRU, Embedding, Input, Activation, 
                         concatenate, GlobalAveragePooling1D, LSTM, 
                         Bidirectional, GlobalMaxPooling1D, SpatialDropout1D,
                         BatchNormalization)
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import get_custom_objects
from keras.preprocessing import text, sequence
from src.config import Config


class NeuralNetworkModel:
    """Neural network model for toxic comment classification"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_matrix = None
        self.tokenizer = None
        
        # Register custom activation
        def swish(x):
            return K.sigmoid(x) * x
        get_custom_objects().update({'swish': Activation(swish)})
    
    def load_embeddings(self, embedding_path: str) -> Dict[str, np.ndarray]:
        """Load word embeddings from file"""
        embeddings_index = {}
        
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                if len(values) > 1:
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
        
        return embeddings_index
    
    def create_embedding_matrix(self, tokenizer: text.Tokenizer, 
                              embeddings_index: Dict[str, np.ndarray]) -> np.ndarray:
        """Create embedding matrix from tokenizer and embeddings"""
        word_index = tokenizer.word_index
        
        # Initialize with random normal distribution
        matrix = np.stack(list(embeddings_index.values()))
        mean, std = matrix.mean(), matrix.std()
        
        embedding_matrix = np.random.normal(mean, std, (len(word_index) + 1, self.config.model.embed_size))
        
        # Fill in known embeddings
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        return embedding_matrix
    
    def prepare_data(self, train_text: pd.DataFrame, valid_text: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare text data for neural network"""
        train_text['comment_text'] = train_text['comment_text'].fillna('nan')
        valid_text['comment_text'] = valid_text['comment_text'].fillna('nan')
        
        train_text_list = list(train_text['comment_text'].values)
        valid_text_list = list(valid_text['comment_text'].values)
        
        self.tokenizer = text.Tokenizer(lower=True, char_level=False, 
                                      num_words=self.config.model.usable_vocab)
        self.tokenizer.fit_on_texts(train_text_list + valid_text_list)
        
        train_tokens = self.tokenizer.texts_to_sequences(train_text_list)
        valid_tokens = self.tokenizer.texts_to_sequences(valid_text_list)
        
        train_seq = sequence.pad_sequences(train_tokens, maxlen=self.config.model.seq_length)
        valid_seq = sequence.pad_sequences(valid_tokens, maxlen=self.config.model.seq_length)
        
        return train_seq, valid_seq, self.tokenizer.word_index
    
    def build_model(self, embedding_matrix: np.ndarray) -> Model:
        """Build neural network architecture"""
        rnn_params = {
            'units': self.config.model.rnn_units,
            'return_sequences': True,
            'recurrent_dropout': self.config.model.recurrent_dropout,
            'dropout': self.config.model.dropout,
            'activation': 'tanh'
        }
        
        inputs = Input(shape=(self.config.model.seq_length,), name='sequence')
        embed = Embedding(len(embedding_matrix), self.config.model.embed_size, 
                         weights=[embedding_matrix], trainable=False)(inputs)
        embed = SpatialDropout1D(0.2)(embed)
        
        # First GRU layer with return_sequences=True
        lstm_1 = Bidirectional(GRU(**rnn_params))(embed)
        lstm_1 = BatchNormalization()(lstm_1)
        
        # Second GRU layer with return_sequences=False
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
        
        model = Model(inputs=[inputs], output=predict)
        optimizer = Adam(lr=self.config.model.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        return model


class NaiveBayesSVM:
    """Naive Bayes SVM model for toxic comment classification"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def tokenize(self, text: str) -> list:
        """Custom tokenizer for NB-SVM"""
        re_tok = re.compile(f'([{string.punctuation}""¨«»®´·º½¾¿¡§£₤''])')
        return re_tok.sub(r' \1 ', text).split()
    
    def probability(self, document, y_i: int, y: np.ndarray) -> np.ndarray:
        """Calculate word probabilities for class y_i"""
        p = document[y == y_i].sum(0)
        return (p + 1) / ((y == y_i).sum() + 1)
    
    def compute_features(self, document, labels: pd.Series) -> Tuple[LogisticRegression, np.ndarray]:
        """Compute NB-SVM features"""
        y = labels.values
        r = np.log(self.probability(document, 1, y) / self.probability(document, 0, y))
        m = LogisticRegression(C=4, dual=True)
        x_nb = document.multiply(r)
        return m.fit(x_nb, y), r
    
    def train_model(self, train_labels: pd.DataFrame, train_text: pd.DataFrame, 
                   valid_text: pd.DataFrame, test_text: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train NB-SVM model"""
        params = {
            'ngram_range': (1, 2),
            'tokenizer': self.tokenize,
            'min_df': 3,
            'max_df': 0.9,
            'strip_accents': 'unicode',
            'use_idf': 1,
            'smooth_idf': 1,
            'sublinear_tf': 1
        }
        
        vectorizer = TfidfVectorizer(**params)
        train_doc = vectorizer.fit_transform(train_text['comment_text'].fillna('nan'))
        valid_doc = vectorizer.transform(valid_text['comment_text'].fillna('nan'))
        test_doc = vectorizer.transform(test_text['comment_text'].fillna('nan'))
        
        valid_ids = valid_text[['id']].copy()
        test_ids = test_text[['id']].copy()
        
        valid_score = np.zeros([valid_text.shape[0], train_labels.shape[1] - 1])
        test_score = np.zeros([test_text.shape[0], train_labels.shape[1] - 1])
        
        for idx in range(train_labels.shape[1] - 1):
            model, result = self.compute_features(train_doc, train_labels.iloc[:, idx + 1])
            valid_score[:, idx] = model.predict_proba(valid_doc.multiply(result))[:, 1]
            test_score[:, idx] = model.predict_proba(test_doc.multiply(result))[:, 1]
        
        valid_score_df = pd.DataFrame(valid_score)
        valid_score_df.columns = list(train_labels.columns)[1:]
        valid_score_df = valid_ids.join(valid_score_df)
        
        test_score_df = pd.DataFrame(test_score)
        test_score_df.columns = list(train_labels.columns)[1:]
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
            'sublinear_tf': 1,
            'max_features': 10000
        }
        word_vectorizer = TfidfVectorizer(**word_params)
        
        # Char-level TF-IDF
        char_params = {
            'analyzer': 'char',
            'ngram_range': (2, 6),
            'token_pattern': r'\w{1,}',
            'stop_words': 'english',
            'strip_accents': 'unicode',
            'sublinear_tf': 1,
            'max_features': 50000
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
        
        valid_ids = valid_text[['id']].copy()
        test_ids = test_text[['id']].copy()
        
        valid_score = np.zeros([valid_text.shape[0], train_labels.shape[1] - 1])
        test_score = np.zeros([test_text.shape[0], train_labels.shape[1] - 1])
        
        for idx in range(train_labels.shape[1] - 1):
            model = LogisticRegression(C=0.1, solver='sag')
            model.fit(train_features, train_labels.iloc[:, idx + 1].values)
            valid_score[:, idx] = model.predict_proba(valid_features)[:, 1]
            test_score[:, idx] = model.predict_proba(test_features)[:, 1]
        
        valid_score_df = pd.DataFrame(valid_score)
        valid_score_df.columns = list(train_labels.columns)[1:]
        valid_score_df = valid_ids.join(valid_score_df)
        
        test_score_df = pd.DataFrame(test_score)
        test_score_df.columns = list(train_labels.columns)[1:]
        test_score_df = test_ids.join(test_score_df)
        
        return valid_score_df, test_score_df


def evaluate_predictions(predictions: pd.DataFrame, actual: pd.DataFrame, 
                        target_columns: list) -> Dict[str, float]:
    """Evaluate model predictions using ROC AUC"""
    results = {}
    overall = 0.0
    
    for col in target_columns:
        if col in predictions.columns and col in actual.columns:
            fpr, tpr, _ = roc_curve(actual[col], predictions[col])
            auc_score = auc(fpr, tpr)
            results[col] = round(auc_score, 4)
            overall += auc_score
    
    results['overall'] = round(overall / len(target_columns), 4)
    return results
