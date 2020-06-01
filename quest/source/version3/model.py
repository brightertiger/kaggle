import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.models import Model
from transformers import BertModel
from transformers.modeling_tf_bert import BertConfig, TFBertModel

MAX_SEQUENCE_LENGTH = 512
bert_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=False)

def BERTModel():
    q_idx = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_idx = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    q_msk = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_msk = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    q_atn = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_atn = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    bert_model = TFBertModel.from_pretrained('bert-base-uncased', config = bert_config)
    q_emb, q_pol = bert_model(q_idx, attention_mask=q_msk, token_type_ids=q_atn)
    a_emb, a_pol = bert_model(a_idx, attention_mask=a_msk, token_type_ids=a_atn)
    q_emb = GlobalAveragePooling1D()(q_emb)
    a_emb = GlobalAveragePooling1D()(a_emb)
    hidden = Concatenate()([q_emb, q_pol, a_emb, a_pol])
    hidden = Dropout(0.2)(hidden)
    output = Dense(30, activation='sigmoid')(hidden)
    model = Model(inputs=[q_idx, q_msk, q_atn, a_idx, a_msk, a_atn], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(lr=1e-5))
    return model