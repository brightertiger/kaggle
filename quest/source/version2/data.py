import re
import pandas as pd
import numpy as np
import tensorflow.keras as keras
from transformers import BertModel, BertTokenizer
from math import floor, ceil

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

def _trim_input_(title, question, answer):
    t_max_len = 30
    q_max_len = 239 
    a_max_len = 239
    max_sequence_length = 512
    t = TOKENIZER.tokenize(title)
    q = TOKENIZER.tokenize(question)
    a = TOKENIZER.tokenize(answer)
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)
    if (t_len+q_len+a_len+4) > max_sequence_length:
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]
    return t, q, a

def _get_ids_(tokens):
    max_seq_length = 512
    token_ids = TOKENIZER.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0]*(max_seq_length - len(token_ids))
    return input_ids

def _get_masks_(tokens):
    max_seq_length = 512
    return [1]*len(tokens)+[0]*(max_seq_length-len(tokens))

def _get_segments_(tokens):
    max_seq_length = 512
    segments=[]
    first_sep=True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == '[ANS]':
            current_segment_id=1
    return segments+[0]*(max_seq_length-len(tokens))

def _convert_to_bert_inputs_(title, question, answer):
    title, question, answer = _trim_input_(title, question, answer)
    tokens = ['[CLS]'] + title + ['[QBODY]'] + question + ['[ANS]'] + answer + ['[SEP]']
    input_ids = _get_ids_(tokens)
    input_masks = _get_masks_(tokens)
    input_segments = _get_segments_(tokens)
    return input_ids, input_masks, input_segments

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
        input_ids, input_masks, input_segments = _convert_to_bert_inputs_(title, question, answer)
        input_ids = np.array(input_ids, dtype=np.int32).reshape(1,-1)
        input_masks = np.array(input_masks, dtype=np.int32).reshape(1,-1)
        input_segments = np.array(input_segments, dtype=np.int32).reshape(1,-1)
        labels = self.label[self.label['qa_id'] == idx].iloc[:,1:]
        labels = np.array(labels, dtype=np.float32).reshape(1,-1)
        return ([input_ids, input_masks, input_segments], labels)

def dataLoader(fold):
    train_data = ModelDataset('train', fold, True)
    valid_data = ModelDataset('valid', min(5,fold), False)
    return train_data, valid_data
