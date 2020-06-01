import pandas as pd
import numpy as np
import re
from keras.layers import Dense, Dropout, GRU, Embedding, Conv1D, MaxPooling1D
from keras.layers import Input, Activation, concatenate, GlobalAveragePooling1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout1D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing import text, sequence
from keras import backend as K
import gc
np.random.seed(2017)

embeddings_index = {}

f = open('../data/data/neural/embed/wiki.ru.vec')

skip = True

for line in f:
    if not skip:
        values = line.rstrip().split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    else:
        skip = False
f.close()

print('Found %s word vectors.' % len(embeddings_index))

matrix = np.stack(embeddings_index.values())
mean, std = matrix.mean(), matrix.std()

VOCAB_TEXT = 195918 + 1
VOCAB_IMAGE = 3063 + 1
EMBED_IMAGE = 64
VOCAB_REGION = 28 + 1
EMBED_REGION = 16
VOCAB_CITY = 1752 + 1
EMBED_CITY = 32
VOCAB_PARAM_1 = 372 + 1
EMBED_PARAM_1 = 32
VOCAB_PARAM_2 = 278 + 1
EMBED_PARAM_2 = 32
VOCAB_PARAM_3 = 1277 + 1
EMBED_PARAM_3 = 32
VOCAB_CATEGORY = 47 + 1
EMBED_CATEGORY = 16
EMBED_NLP = 300
EMBED_CAT = 25
REGULAR = 1e-5

def define_model(matrix): 
    params = {}
    params['embeddings_regularizer'] = l2(REGULAR)
    # description
    text = Input(shape=(100,), name='description')
    embed_text = Embedding(VOCAB_TEXT, EMBED_NLP, weights=[matrix], trainable=False)(text)
    embed_text = SpatialDropout1D(0.3)(embed_text)
    convolve = Conv1D(128, kernel_size=3, activation='relu')(embed_text)
    convolve = Conv1D(256, kernel_size=3, activation='relu')(convolve)
    convolve = GlobalMaxPooling1D()(convolve)    
    # region
    region = Input(shape=(1,), name='region')
    embed_region = Embedding(VOCAB_REGION, EMBED_REGION, **params)(region)
    embed_region = Flatten()(embed_region)    
    # city
    city = Input(shape=(1,), name='city')
    embed_city = Embedding(VOCAB_CITY, EMBED_CITY, **params)(city)
    embed_city = Flatten()(embed_city)
    # param - 1
    param_1 = Input(shape=(1,), name='param_1')
    embed_param_1 = Embedding(VOCAB_PARAM_1, EMBED_PARAM_1, **params)(param_1)
    embed_param_1 = Flatten()(embed_param_1)
    # param - 2
    param_2 = Input(shape=(1,), name='param_2')
    embed_param_2 = Embedding(VOCAB_PARAM_2, EMBED_PARAM_2, **params)(param_2)
    embed_param_2 = Flatten()(embed_param_2)
    # param - 3
    param_3 = Input(shape=(1,), name='param_3')
    embed_param_3 = Embedding(VOCAB_PARAM_3, EMBED_PARAM_3, **params)(param_3)
    embed_param_3 = Flatten()(embed_param_3)
    # category
    category = Input(shape=(1,), name='category')
    embed_category = Embedding(VOCAB_CATEGORY, EMBED_CATEGORY, **params)(category)
    embed_category = Flatten()(embed_category)
    # image
    image = Input(shape=(1,), name='image')
    embed_image = Embedding(VOCAB_IMAGE, EMBED_IMAGE, **params)(image)
    embed_image = Flatten()(embed_image)
    # numeric 
    numeric = Input(shape=(12,))
    # concatenate
    concat =  [convolve, embed_region, embed_city]
    concat += [embed_param_1, embed_param_2, embed_param_3]
    concat += [embed_category, embed_image]
    dense = concatenate(concat)
    dense = Dropout(0.3)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(512, activation='relu', kernel_regularizer=l2(0.00001))(dense)
    dense = Dropout(0.3)(dense)
    dense = concatenate([dense,numeric])
    dense = BatchNormalization()(dense)
    dense = Dense(256, activation='relu', kernel_regularizer=l2(0.00001))(dense)
    dense = Dropout(0.3)(dense)
    predict = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001))(dense)
    # compile
    inputs =  [text, numeric, region, city]
    inputs += [param_1, param_2, param_3]
    inputs += [category, image]
    model = Model(inputs=inputs, output=predict)
    optimizer = Adam(lr=1e-3, clipnorm=0.75)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    return model

def dataflow(train_text, valid_text, length):
    train_text = list(train_text.fillna('nan').values)
    valid_text = list(valid_text.fillna('nan').values)
    tokenizer = text.Tokenizer(lower=True, char_level=False, oov_token='none')
    tokenizer.fit_on_texts(train_text + valid_text)
    word_index = tokenizer.word_index
    embedding_matrix = None
    intersect = 0
    global mean, std
    embedding_matrix = np.random.normal(mean, std, (len(word_index) + 1, EMBED_NLP))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            intersect += 1
    print('total words:', len(word_index))
    print('intersection:', intersect)
    train_text = tokenizer.texts_to_sequences(train_text)
    valid_text = tokenizer.texts_to_sequences(valid_text)
    train_text = sequence.pad_sequences(train_text, maxlen=length)
    valid_text = sequence.pad_sequences(valid_text, maxlen=length)
    return train_text, valid_text, embedding_matrix

def callbacks(mode):
    stop = EarlyStopping('val_loss', patience=5, mode="min")
    logger = CSVLogger('../data/data/neural/model_2/logger_{}.log'.format(mode))
    best = ModelCheckpoint('../data/data/neural/model_2/weights_{}.h5'.format(mode), save_best_only=True)
    anneal = ReduceLROnPlateau(factor=0.5, patience=2)
    return [stop, logger, best, anneal]

feat_encode   = ['region','city','param_1','param_2','param_3', 'category_name','image_top_1']
feat_numeric  = ['count_1', 'count_3', 'count_4','count_5']
feat_numeric += ['relative_1','relative_2','relative_5','relative_6']
feat_numeric += ['renewal_1', 'avg_days_up_user']
feat_numeric += ['price','item_seq_number']

def execute(mode):
    train_data = pd.read_csv('../data/data/neural/dataset/train_{}.csv'.format(mode))
    valid_data = pd.read_csv('../data/data/neural/dataset/valid_{}.csv'.format(mode))
    train_data = train_data.drop('item_id', axis=1)
    valid_data = valid_data.drop('item_id', axis=1)
    params = []
    params += [train_data['text'], valid_data['text']]
    train_text, valid_text, embed_text  = dataflow(*params, length=100)
    train_inputs = [train_text]
    train_inputs += [train_data[feat_numeric].fillna(-1.).values]
    for feat in feat_encode:
        train_inputs += [train_data[feat].values]
    valid_inputs = [valid_text]
    valid_inputs += [valid_data[feat_numeric].fillna(-1.).values]
    for feat in feat_encode:
        valid_inputs += [valid_data[feat].values]
    train_labels = train_data['deal_probability'].values
    valid_labels = valid_data['deal_probability'].values
    params = {}
    params['x'] = train_inputs
    params['y'] = np.array(train_labels)
    params['validation_data'] = (valid_inputs, np.array(valid_labels))
    params['batch_size'] = 1024
    params['verbose'] = 1
    params['callbacks'] = callbacks(mode)
    params['epochs'] = 25
    model = define_model(embed_text)
    model.fit(**params)
    del model
    gc.collect()
    K.clear_session()
    return None

execute(1)
execute(2)
execute(3)
execute(4)
execute(5)