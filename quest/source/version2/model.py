import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.models import Model
from transformers import BertModel
from transformers.modeling_tf_bert import BertConfig, TFBertModel

MAX_SEQUENCE_LENGTH = 512
bert_config = BertConfig(unk_token="[QBODY]", pad_token="[ANS]")
bert_config = bert_config.from_pretrained('bert-base-uncased', output_hidden_states=True)

def BERTModel():
    input_ids = Input((MAX_SEQUENCE_LENGTH), dtype = tf.int32, name = 'input_ids')
    input_mask = Input((MAX_SEQUENCE_LENGTH), dtype = tf.int32, name = 'input_masks')
    input_segments = Input((MAX_SEQUENCE_LENGTH), dtype = tf.int32, name = 'input_segments')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased',config = bert_config)
    sequence_output, pooler_output, hidden_states = bert_model([input_ids, input_mask, input_segments])
    hidden = GlobalAveragePooling1D()(sequence_output)
    hidden = Concatenate()([pooler_output, hidden])
    hidden = Dropout(0.2)(hidden)
    output = Dense(30, activation='sigmoid', name='output')(hidden)
    model = Model(inputs=[input_ids, input_mask, input_segments], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(lr=1e-5))
    return model