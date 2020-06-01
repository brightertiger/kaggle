import pandas as pd
import numpy as np
import re
import gc
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
from sklearn.metrics import mean_squared_error

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

def define_model(): 
    params = {}
    params['embeddings_regularizer'] = l2(REGULAR)
    # description
    text = Input(shape=(100,), name='description')
    embed_text = Embedding(VOCAB_TEXT, EMBED_NLP)(text)
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

def dataflow(train_text, valid_text, score_text, length):
    train_text = list(train_text.fillna('nan').values)
    valid_text = list(valid_text.fillna('nan').values)
    tokenizer = text.Tokenizer(lower=True, char_level=False, oov_token='none')
    tokenizer.fit_on_texts(train_text + valid_text)
    score_text = tokenizer.texts_to_sequences(score_text)
    score_text = sequence.pad_sequences(score_text, maxlen=length)
    return score_text

feat_encode   = ['region','city','param_1','param_2','param_3', 'category_name','image_top_1']
feat_numeric  = ['count_1', 'count_3', 'count_4','count_5']
feat_numeric += ['relative_1','relative_2','relative_5','relative_6']
feat_numeric += ['renewal_1', 'avg_days_up_user']
feat_numeric += ['price','item_seq_number']

def execute(mode):
    train_data = pd.read_csv('../data/data/neural/dataset/train_{}.csv'.format(mode))
    valid_data = pd.read_csv('../data/data/neural/dataset/valid_{}.csv'.format(mode))
    score_data = pd.read_csv('../data/data/neural/dataset/valid_{}.csv'.format(mode))
    train_data = train_data.drop('item_id', axis=1)
    valid_data = valid_data.drop('item_id', axis=1)
    driver = score_data[['item_id']].copy()
    params = []
    params += [train_data['text'], valid_data['text'], score_data['text'].fillna('nan')]
    score_text  = dataflow(*params,length=100)
    score_inputs = [score_text]
    score_inputs += [score_data[feat_numeric].fillna(-1.).values]
    for feat in feat_encode:
        score_inputs += [score_data[feat].values]
    model = define_model()
    model.load_weights('../data/data/neural/model_2/weights_{}.h5'.format(mode))
    scores = model.predict(score_inputs, batch_size=512)
    scores = pd.DataFrame(scores, columns=['deal_probability'])
    scores = driver.join(scores)
    scores.to_csv('../data/data/neural/score_2/valid_{}.csv'.format(mode), index=False)
    print('{} mode scoring finished...'.format(mode))
    return None

for i in range(5):
    execute(i+1)

def execute(mode):
    train_data = pd.read_csv('../data/data/neural/dataset/train_{}.csv'.format(mode))
    valid_data = pd.read_csv('../data/data/neural/dataset/valid_{}.csv'.format(mode))
    score_data = pd.read_csv('../data/data/neural/test_data.csv')
    train_data = train_data.drop('item_id', axis=1)
    valid_data = valid_data.drop('item_id', axis=1)
    driver = score_data[['item_id']].copy()
    params = []
    params += [train_data['text'], valid_data['text'], score_data['text'].fillna(' ')]
    score_text  = dataflow(*params,length=100)
    score_inputs = [score_text]
    score_inputs += [score_data[feat_numeric].fillna(-1.).values]
    for feat in feat_encode:
        score_inputs += [score_data[feat].values]
    model = define_model()
    model.load_weights('../data/data/neural/model_2/weights_{}.h5'.format(mode))
    scores = model.predict(score_inputs, batch_size=512)
    scores = pd.DataFrame(scores, columns=['deal_probability'])
    scores = driver.join(scores)
    scores.to_csv('../data/data/neural/score_2/test_{}.csv'.format(mode), index=False)
    print('{} mode scoring finished...'.format(mode))
    return None

for i in range(5):
    execute(i+1)

dataset = pd.DataFrame([])
for i in range(5):
    i += 1
    temp = pd.read_csv('../data/data/neural/score_2/valid_{}.csv'.format(i))
    dataset = dataset.append(temp)

dataset.to_csv('../data/insample/scores/cnn_2.csv', index=False)
actuals = pd.read_csv('../data/download/train.csv', usecols=['item_id','deal_probability'])
actuals = actuals.merge(dataset, on='item_id')
np.sqrt(mean_squared_error(actuals['deal_probability_x'], actuals['deal_probability_y']))

dataset = pd.DataFrame([])
for i in range(5):
    i += 1
    temp = pd.read_csv('../data/data/neural/score_2/test_{}.csv'.format(i))
    dataset = dataset.append(temp)

dataset = dataset.groupby(['item_id'])['deal_probability'].mean().reset_index()
dataset.to_csv('../data/outsample/scores/cnn_2.csv', index=False)