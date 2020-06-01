import pandas as pd
import numpy as np
import tensorflow.keras as keras
from transformers import BertModel, BertTokenizer
from math import floor, ceil
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

def _array_(x):
    return np.array(x, dtype=np.int32).reshape(1,-1)

def _ids_(string):
    inputs = TOKENIZER.tokenize(string)
    if len(inputs) >= 500:
        inputs = inputs[:350] + inputs[-150:]
    inputs = ['[CLS]'] + inputs + ['[SEP]']
    lengths = len(inputs)
    padding = 512 - lengths
    input_ids =  TOKENIZER.convert_tokens_to_ids(inputs)
    input_ids = input_ids + [0] * padding
    input_masks = [1] * lengths + [0] * padding
    input_segments = [1] * lengths + [0] * padding
    input_ids = _array_(input_ids)
    input_masks = _array_(input_masks)
    input_segments = _array_(input_segments)
    return input_ids, input_masks, input_segments

def _tokens_(title, question, answer, max_len):
    q_idx, q_msk, q_atn = _ids_(title + ' ' + question)
    a_idx, a_msk, a_atn = _ids_(answer)
    return [q_idx, q_msk, q_atn, a_idx, a_msk, a_atn]

class ModelDataset(keras.utils.Sequence):
    
    def __init__(self, mode, fold, shuffle):
        prefix = '../../data/split/'
        if fold != 6 or (fold == 6 and mode == 'valid'):
            self.label = pd.read_csv(prefix + 'label_{}_{}.csv'.format(mode, fold))
            self.data = pd.read_csv(prefix + 'data_{}_{}.csv'.format(mode, fold))
            self.text =  pd.read_csv(prefix + 'text_{}_{}.csv'.format(mode, fold))
            self.index = self.label['qa_id'].tolist()
            self.looper = list(np.arange(len(self.index)))
        else:
            train_label = pd.read_csv(prefix + 'label_train_1.csv')
            train_data = pd.read_csv(prefix + 'data_train_1.csv')
            train_text =  pd.read_csv(prefix + 'text_train_1.csv')
            valid_label = pd.read_csv(prefix + 'label_valid_1.csv')
            valid_data = pd.read_csv(prefix + 'data_valid_1.csv')
            valid_text =  pd.read_csv(prefix + 'text_valid_1.csv')
            self.label = train_label.append(valid_label).reset_index(drop=True)
            self.data = train_data.append(valid_data).reset_index(drop=True)
            self.text = train_text.append(valid_text).reset_index(drop=True)
            self.index = self.label['qa_id'].tolist()
            self.looper = list(np.arange(len(self.index)))
        self.shuffle = shuffle
        return None

    def __len__(self):
        return len(self.index)
    
    def on_epoch_end(self):
        if self.shuffle: 
            np.random.shuffle(self.looper)
        return None
    
    def __getitem__(self, index):
        idx = self.looper[index]
        idx = self.index[idx]
        title = self.text[self.text['qa_id'] == idx].iat[0,1]
        question = self.text[self.text['qa_id'] == idx].iat[0,2]
        answer = self.text[self.text['qa_id'] == idx].iat[0,3]
        inputs = _tokens_(title, question, answer, 512)
        labels = self.label[self.label['qa_id'] == idx].iloc[:,1:]
        labels = np.array(labels, dtype=np.float32).reshape(1,-1)
        return (inputs, labels)

def dataLoader(fold):
    train_data = ModelDataset('train', fold, True)
    valid_data = ModelDataset('valid', min(fold,5), False)
    return train_data, valid_data
