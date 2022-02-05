from random import sample
from typing import Dict, Any
from kashgari import embeddings
import numpy as np
import pickle

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import optimizers, losses
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn import metrics as sklearn_metrics

from bert4keras.models import build_transformer_model
from transformers import TFMPNetModel, TFBertModel, BertModel


class Scaffold_Model():
    def __init__(self, embedding_path, embedding_type, output_dims, feature_D):
        self.vocab_path = f'{embedding_path}/vocab.txt'
        self.embedding_type = embedding_type
        self.output_dims = output_dims
        self.feature_D = feature_D

        self.id2label_list, self.label2id_list = [], []
        self.token2idx = {}
        with open(self.vocab_path, 'r', encoding='utf-8') as reader:
            for line in reader.readlines():
                token = line.strip()
                self.token2idx[token] = len(self.token2idx)
        
        if embedding_type == 'mpnet':
            self.embedding = TFMPNetModel.from_pretrained(embedding_path)
        if embedding_type == 'bert_hf':
            # self.embedding = build_transformer_model(config_path=f'{embedding_path}/bert_config.json',
            #                                      checkpoint_path=f'{embedding_path}/bert_model.ckpt',
            #                                      application='encoder',
            #                                      return_keras_model=True)
            self.embedding = TFBertModel.from_pretrained(embedding_path, from_pt=True)

        for layer in self.embedding.layers:
            layer.trainable = False
        
        self.model = self.build_model_arc()
        self.model.compile(loss=losses.SparseCategoricalCrossentropy(),
                           loss_weights=[1, 0.1, 0.05],
                           optimizer=optimizers.Adam(),
                           metrics=None)
        # self.model.summary()

    def encode_seq(self, seqs):
        max_len = 0
        seq_encodes = []
        for i in range(len(seqs)):
            seq_encode = [self.token2idx[word] for word in seqs[i]]
            max_len = max(max_len, len(seq_encode))
            seq_encodes.append(seq_encode)
        return pad_sequences(seq_encodes, padding="post", truncating="post", value=0, maxlen=max_len, dtype='int32')
    
    # def init_label(self, y0, y1, y2):
    #     label2id, id2label = {}, {}
    #     for i in range(1, 3):
    #         for idx, label in enumerate(set(eval(f'y{i}'))):
    #             label2id[label] = idx
    #             id2label[idx] = label
    #         self.id2label_list.append(id2label)
    #         self.label2id_list.append(label2id)
    
    def encode_label(self, labels, task_num):
        return np.array([int(label) for label in labels])
    
    def fusion_layer(self, tensor, features_tensor):
        if self.embedding_type == 'mpnet':
            tensor = self.embedding(tensor).hidden_states[1]
        if self.embedding_type == 'bert_hf':
            tensor = self.embedding(tensor).hidden_states[1]

        tensor = L.Bidirectional(L.LSTM(units=128, return_sequences=True))(tensor)
        tensor = L.Dropout(rate=0.1)(tensor)

        features_tensor = L.Bidirectional(L.LSTM(units=32, return_sequences=True))(features_tensor)
        features_tensor = L.Dropout(rate=0.1)(features_tensor)

        tensor = L.Concatenate(axis=-1)([tensor, features_tensor])
        tensor = L.Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu')(tensor)

        query_value_attention_seq = L.MultiHeadAttention(
            num_heads=2, key_dim=2, dropout=0.5
        )(tensor, tensor)

        query_encoding = L.GlobalMaxPool1D()(tensor)
        query_value_attention = L.GlobalMaxPool1D()(query_value_attention_seq)
        input_tensor = L.Concatenate()([query_encoding, query_value_attention])

        input_tensor = L.Dropout(rate=0.1)(input_tensor)

        return input_tensor
    
    def bilstm_att(self, tensor):
        if self.embedding_type == 'mpnet':
            tensor = self.embedding(tensor).hidden_states[1]
        if self.embedding_type == 'bert_hf':
            tensor = self.embedding(tensor).hidden_states[2]

        tensor = L.Bidirectional(L.LSTM(units=128, return_sequences=True))(tensor)
        tensor = L.Dropout(rate=0.1)(tensor)

        tensor_encoding = L.GlobalMaxPool1D()(tensor)
        tensor_att_value = L.GlobalMaxPool1D()(L.Attention()([tensor, tensor]))
        input_tensor = L.Concatenate(axis=-1)([tensor_encoding, tensor_att_value])

        input_tensor = L.Dropout(rate=0.1)(input_tensor)

        return input_tensor
        
    def build_model_arc(self):
        l2_reg = tf.keras.regularizers.L2(0.01)

        tensor0 = tf.keras.Input(shape=(None, ), name="data0", dtype='int32')
        tensor1 = tf.keras.Input(shape=(None, ), name="data1", dtype='int32')
        tensor2 = tf.keras.Input(shape=(None, ), name="data2", dtype='int32')

        features = tf.keras.Input(shape=(None, self.feature_D), name="features")
        features_tensor = features

        input_tensor0 = self.fusion_layer(tensor0, features_tensor)
        input_tensor1 = self.bilstm_att(tensor1)
        input_tensor2 = self.bilstm_att(tensor2)

        activations = ['softmax', 'sigmoid', 'softmax']
        inputs = [input_tensor0, input_tensor1, input_tensor2]

        output_tensor = [L.Dense(self.output_dims[i], activation=activations[i], name="output" + str(i),
                                    kernel_regularizer=l2_reg)(inputs[i])
                            for i in range(3)]

        return tf.keras.Model(inputs=[tensor0, tensor1, tensor2, features], outputs=output_tensor)
    
    def fit(self, x0, y0, x1, y1, x2, y2, f, batch_size, epoch):
        data, labels = {}, {}
        for i in range(3):
            data[f'data{i}'] = self.encode_seq(eval(f'x{i}'))
            labels[f'output{i}'] = self.encode_label(eval(f'y{i}'), i)
            
        dim_x = len(data['data0'][0])
        # 特征对齐，统一数据和特征的维度
        features = pad_sequences(f, maxlen=dim_x, dtype='int32', padding='post',truncating='post', value=0)
        data['features'] = features

        self.model.fit(data, labels, validation_split=0.1, batch_size=batch_size, epochs=epoch)
    
    def evaluate(self, x0, y0, x1, y1, x2, y2, f, batch_size):
        data, y_data = {}, []
        for i in range(3):
            data[f'data{i}'] = self.encode_seq(eval(f'x{i}'))
            y_data.append(self.encode_label(eval(f'y{i}'), i))

        dim_x = len(data['data0'][0])
        # 特征对齐，统一数据和特征的维度
        features = pad_sequences(f, maxlen=dim_x, dtype='int32', padding='post',truncating='post', value=0)
        data['features'] = features

        len0, len1, len2 = len(data['data0']), len(data['data1']), len(data['data2'])
        max_sample_len = max(len0, len1, len2)
        def build_virtual_input(true_inputs):
            sample_len = len(true_inputs)
            return np.array(true_inputs.tolist()+[true_inputs[0] for _ in range(max_sample_len-sample_len)])


        reports = []
        # 由于predict需要一个对齐的输入, 这里直接预测3次，如果不要输出3个结果也行
        virtual_data0 = build_virtual_input(data['data0'])
        virtual_data1 = build_virtual_input(data['data1'])
        virtual_data2 = build_virtual_input(data['data2'])
        virtual_f = build_virtual_input(features)
        inputs_data_list = [
            {'data0':data['data0'], 'data1':virtual_data1[:len0], 'data2':virtual_data1[:len0], 'features':features},
            {'data0':virtual_data0[:len1], 'data1':data['data1'], 'data2':virtual_data2[:len1], 'features':virtual_f[:len1]},
            {'data0':virtual_data0[:len2], 'data1':virtual_data1[:len2], 'data2':data['data2'], 'features':virtual_f[:len2]}
        ]
        for i in range(1):
            pred = self.model.predict(inputs_data_list[i], batch_size=batch_size)
            pred_max = pred[i].argmax(-1)
            print(sklearn_metrics.classification_report(y_data[i],
                            pred_max, output_dict=False, digits=4))
            original_report= sklearn_metrics.classification_report(y_data[i],
                            pred_max, output_dict=True, digits=4)

            reports.append({
                'detail': original_report,
            })

            # pickle.dump(sklearn_metrics.confusion_matrix(y_data[i], pred_max),
            #     open(f'./reference/scaffold_confusion_matrix_task{i}.pkl', 'wb'))

        return reports
            