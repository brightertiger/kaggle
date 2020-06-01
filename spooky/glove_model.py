import pandas as pd
import numpy as np
import keras
from keras.layers import Dense, GlobalAveragePooling1D, Embedding
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import Adam
import os
pd.options.mode.chained_assignment = None
np.random.seed(2017)

train_data = pd.read_csv('../../data/spooky-author/download/train.csv')
test_data = pd.read_csv('../../data/spooky-author/download/test.csv')
mapper = {'EAP':0, 'HPL':1, 'MWS':2}
train_data['author'] = train_data['author'].map(lambda x : mapper[x])

train_texts = train_data['text'].values.tolist()
score_texts = test_data['text'].values.tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts + score_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(score_texts)
word_index = tokenizer.word_index
print('words:', len(word_index))
train_seq = pad_sequences(train_sequences, maxlen=90)
score_seq = pad_sequences(test_sequences, maxlen=90)
labels = to_categorical(train_data['author'].values.tolist())
print('data:', train_seq.shape, score_seq.shape)
print('label:', labels.shape)

indices = np.arange(train_seq.shape[0])
np.random.shuffle(indices)
train_seq = train_seq[indices]
labels = labels[indices]
nb_validation_samples = int(0.33 * train_seq.shape[0])
x_train = train_seq[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = train_seq[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}

f = open(os.path.join('../../src/pip/glove/glove.6B.50d.txt'))

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, 50))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

del embeddings_index

embedding_layer = Embedding(len(word_index) + 1, 50, weights=[embedding_matrix], input_length=90, trainable=True)

model = Sequential()
model.add(embedding_layer)
model.add(GlobalAveragePooling1D())
model.add(Dense(3, activation='softmax'))
optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

params = {}
params['validation_data'] = (x_val, y_val)
params['batch_size'] = 8
params['epochs'] = 3
params['callbacks'] = [EarlyStopping(patience=2, monitor='val_loss')]
model.optimizer.lr = 0.0001
model.fit(x_train, y_train, **params, )

params = {}
params['validation_data'] = (x_val, y_val)
params['batch_size'] = 8
params['epochs'] = 10
params['callbacks'] = [EarlyStopping(patience=2, monitor='val_loss')]
model.optimizer.lr = 0.001
model.fit(x_train, y_train, **params)

params = {}
params['validation_data'] = (x_val, y_val)
params['batch_size'] = 8
params['epochs'] = 10
params['callbacks'] = [EarlyStopping(patience=2, monitor='val_loss')]
model.optimizer.lr = 0.0005
model.fit(x_train, y_train, **params)

train_predict = pd.DataFrame(model.predict_proba(train_seq), columns=['k1','k2','k3'])
train_class = pd.DataFrame(model.predict_classes(train_seq), columns=['keras'])
test_predict = pd.DataFrame(model.predict_proba(score_seq), columns=['k1','k2','k3'])
test_class = pd.DataFrame(model.predict_classes(score_seq), columns=['keras'])
train_keras = pd.concat([train_data[['id']], train_predict, train_class], axis=1)
test_keras = pd.concat([test_data[['id']], test_predict, test_class], axis=1)
print('train',train_keras.shape)
print('test', test_keras.shape)
train_keras.to_csv('../../data/spooky-author/data/train_nn_score.csv', index=False)
test_keras.to_csv('../../data/spooky-author/data/test_nn_score.csv', index=False)