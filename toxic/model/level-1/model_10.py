import pandas as pd
import numpy as np
import re
from keras.layers import Dense, Dropout, GRU, Embedding 
from keras.layers import Input, Activation, concatenate, GlobalAveragePooling1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout
from keras.layers import Conv1D, MaxPooling1D, CuDNNGRU
from keras.layers.core import SpatialDropout1D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils, get_custom_objects
from keras.preprocessing import text, sequence
from keras_contrib.callbacks import SnapshotCallbackBuilder
from multiprocessing import Pool

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})

SEQ_LENGTH = 200
EMBED_SIZE = 200
VOCAB = 172289
USABLE = 100000
np.random.seed(2017)

def define_model(matrix):
    rnn = {}
    rnn['units'] = 50
    rnn['return_sequences'] = True
    inputs = Input(shape=(SEQ_LENGTH,), name='sequence')
    embed = Embedding(VOCAB,EMBED_SIZE, weights=[matrix], trainable=False)(inputs)
    embed = SpatialDropout1D(0.4)(embed)
    lstm = Bidirectional(CuDNNGRU(**rnn))(embed)
    max_pool = GlobalMaxPooling1D()(lstm)
    avg_pool = GlobalAveragePooling1D()(lstm)
    lstm = Dropout(0.2)(lstm)
    conv = Conv1D(64,4)(lstm)
    conv = Conv1D(128,6)(conv)
    conv_pool = GlobalMaxPooling1D()(conv)
    pool = concatenate([max_pool, avg_pool,conv_pool])
    pool = BatchNormalization()(pool)
    pool = Dropout(0.3)(pool)
    dense = Dense(256, activation='swish')(pool)
    dense = Dropout(0.2)(dense)
    dense = Dense(256, activation='swish')(pool)
    dense = Dropout(0.2)(dense)
    predict = Dense(6, activation='sigmoid')(dense)
    model = Model(inputs=[inputs], output=predict)
    optimizer = Adam(lr=1e-3, clipnorm=1.)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model

embeddings_index = {}
f = open('../data/download/glove.twitter.27B.200d.txt')
skip = False

for line in f:
    if not skip:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    else:
        skip = False
f.close()

print('Found %s word vectors.' % len(embeddings_index))

matrix = np.stack(embeddings_index.values())
mean, std = matrix.mean(), matrix.std()

def dataflow(train_text, valid_text):
    train_text['comment_text'] = train_text['comment_text'].fillna('nan')
    valid_text['comment_text'] = valid_text['comment_text'].fillna('nan')
    train_text = list(train_text['comment_text'].values)
    valid_text = list(valid_text['comment_text'].values)
    tokenizer = text.Tokenizer(lower=True, char_level=False, num_words=USABLE)
    tokenizer.fit_on_texts(train_text + valid_text)
    word_index = tokenizer.word_index
    intersect = 0
    embedding_matrix = np.random.normal(mean, std, (len(word_index) + 1, EMBED_SIZE))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            intersect += 1
    train_token = tokenizer.texts_to_sequences(train_text)
    valid_token = tokenizer.texts_to_sequences(valid_text)
    train_seq = sequence.pad_sequences(train_token, maxlen=SEQ_LENGTH)
    valid_seq = sequence.pad_sequences(valid_token, maxlen=SEQ_LENGTH)
    return train_seq, valid_seq, embedding_matrix

def callbacks(suffix):
    stop = EarlyStopping('val_loss', patience=12, mode="min")
    snap = SnapshotCallbackBuilder(30,3,1e-3)
    snap = snap.get_callbacks('model_{}'.format(suffix))
    logger = CSVLogger('../data/data/source_3/model_3/logger_{}.log'.format(suffix))
    return snap + [stop, logger]

def execute(mode):
    mode += 1 
    train_text = pd.read_csv('../data/data/source_3/train/train_data_{}.csv'.format(mode))
    train_label = pd.read_csv('../data/data/source_3/train/train_labels_{}.csv'.format(mode))
    valid_text = pd.read_csv('../data/data/source_3/train/test_data_{}.csv'.format(mode))
    valid_label = pd.read_csv('../data/data/source_3/train/test_labels_{}.csv'.format(mode))
    train_text, valid_text, embedding_matrix = dataflow(train_text, valid_text)
    params = {}
    params['x'] = train_text
    params['y'] = np.array(train_label.iloc[:,1:])
    params['validation_data'] = (valid_text, np.array(valid_label.iloc[:,1:]))
    params['batch_size'] = 256
    params['epochs'] = 30
    params['verbose'] = 0
    params['callbacks'] = callbacks(mode)
    model = define_model(embedding_matrix)
    model.fit(**params)
    print('executed model:', mode)
    return None

for i in range(9):
    execute(i)