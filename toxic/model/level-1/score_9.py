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
from sklearn.metrics import roc_curve, auc

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})

SEQ_LENGTH = 200
EMBED_SIZE = 200
VOCAB = 172289
USABLE = 80000
np.random.seed(2017)

def define_model(matrix):
    rnn = {}
    rnn['units'] = 50
    rnn['return_sequences'] = True
    inputs = Input(shape=(SEQ_LENGTH,), name='sequence')
    embed = Embedding(VOCAB,EMBED_SIZE, weights=[matrix], trainable=False)(inputs)
    embed = SpatialDropout1D(0.4)(embed)
    lstm_1 = Bidirectional(CuDNNGRU(**rnn))(embed)
    lstm_1 = BatchNormalization()(lstm_1)
    rnn['return_sequences'] = False
    lstm_2 = Bidirectional(CuDNNGRU(**rnn))(embed)
    lstm_2 = BatchNormalization()(lstm_2)
    max_pool = GlobalMaxPooling1D()(lstm_1)
    avg_pool = GlobalAveragePooling1D()(lstm_1)
    pool = concatenate([lstm_2, max_pool, avg_pool])
    pool = BatchNormalization()(pool)
    lstm = Dropout(0.3)(pool)
    dense = Dense(256, activation='swish')(lstm)
    dense = Dropout(0.2)(dense)
    dense = Dense(256, activation='swish')(dense)
    dense = Dropout(0.2)(dense)
    predict = Dense(6, activation='sigmoid')(dense)
    model = Model(inputs=[inputs], output=predict)
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
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

def dataflow(train_text, valid_text, score_text):
    train_text['comment_text'] = train_text['comment_text'].fillna('nan')
    valid_text['comment_text'] = valid_text['comment_text'].fillna('nan')
    score_text['comment_text'] = score_text['comment_text'].fillna('nan')
    train_text = list(train_text['comment_text'].values)
    valid_text = list(valid_text['comment_text'].values)
    score_text = list(score_text['comment_text'].values)
    tokenizer = text.Tokenizer(lower=True, char_level=False, num_words=USABLE)
    tokenizer.fit_on_texts(train_text + valid_text)
    word_index = tokenizer.word_index
    intersect = 0
    embedding_matrix = np.zeros((len(word_index) + 1, EMBED_SIZE))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            intersect += 1
    score_token = tokenizer.texts_to_sequences(score_text)
    score_seq = sequence.pad_sequences(score_token, maxlen=SEQ_LENGTH)
    return score_seq, embedding_matrix

def score_model(mode):
    train_text = pd.read_csv('../data/data/source_3/train/train_data_{}.csv'.format(mode))
    valid_text = pd.read_csv('../data/data/source_3/train/test_data_{}.csv'.format(mode))
    score_text = pd.read_csv('../data/data/source_3/score/score_data.csv')
    score_data = score_text[['id']]
    score_text, embedding_matrix = dataflow(train_text, valid_text, score_text)
    model = define_model(embedding_matrix)
    path = './weights/model_{}-Best.h5'.format(mode)
    model.load_weights(path)
    scores = model.predict(score_text, batch_size=256)
    scores = pd.DataFrame(scores)
    scores.columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    scores = score_data.join(scores)
    return scores

submit = pd.DataFrame([])

for idx in range(9):
    submit = submit.append(score_model(idx+1))
    print('{} model scored...'.format(idx+1))
    
submit = submit.groupby('id').mean().reset_index()
submit.to_csv('../data/submit/model_10.csv', index=False)

def score_model(mode):
    train_text = pd.read_csv('../data/data/source_3/train/train_data_{}.csv'.format(mode))
    valid_text = pd.read_csv('../data/data/source_3/train/test_data_{}.csv'.format(mode))
    score_text = pd.read_csv('../data/data/source_3/train/test_data_{}.csv'.format(mode))
    labels = pd.read_csv('../data/data/source_3/train/test_labels_{}.csv'.format(mode))
    score_data = score_text[['id']]
    score_text, embedding_matrix = dataflow(train_text, valid_text, score_text)
    model = define_model(embedding_matrix)
    path = './weights/model_{}-Best.h5'.format(mode)
    model.load_weights(path)
    scores = model.predict(score_text, batch_size=256)
    scores = pd.DataFrame(scores)
    scores.columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    scores = score_data.join(scores)
    return scores, labels

submit = pd.DataFrame([])
labels = pd.DataFrame([])
for idx in range(9):
    submit_, labels_ = score_model(idx+1)
    submit = submit.append(submit_)
    labels = labels.append(labels_)
    print('{} model scored...'.format(idx+1))
    
submit.to_csv('../data/model/model_10.csv', index=False)

models = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
evaluate = 0.

for subset in models:
    predict = submit[subset]
    actual = labels[subset]
    fpr, tpr, threshold = roc_curve(actual, predict)
    metric = round(2*auc(fpr, tpr)-1, 4)
    print('label:', subset, ':', metric)
    evaluate += metric
    
print('overall:', round(evaluate/6, 4))