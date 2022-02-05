from typing import Dict, Any

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from features.features_layers import FeaturesFusion

from kashgari_local.abc_feature_model import ABCClassificationModel
from kashgari.layers import L


class RCNN_Att_Model(ABCClassificationModel):
    def __init__(self, embedding, **params):
        super().__init__(embedding, task_num=params['task_num'])
        # self.path = params["path"]
        # self.label = params["label"]
        self.feature_D = params["feature_D"]

    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict

        activation_function list:
        {softmax, elu, selu, softplus, softsign, swish,
        relu, gelu, tanh, sigmoid, exponential,
        hard_sigmoid, linear, serialize, deserialize, get}
        """
        return {
            'layer_bilstm1': {
                'units': 128,
                'return_sequences': True
            },
            'layer_time_distributed': {},
            'layer_output1': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        """
        build model architectural

        BiLSTM + Convolution + Attention
        """
        l1_reg = tf.keras.regularizers.l1(0.01)
        l2_reg = tf.keras.regularizers.L2(0.01)

        features = tf.keras.Input(shape=(None, self.feature_D), name="features")
        features_tensor = features
        features_tensor = L.Bidirectional(L.LSTM(units=32, return_sequences=True))(features_tensor)
        features_tensor = L.Dropout(rate=0.1)(features_tensor)

        if self.task_num == 1:
            output_dim = self.label_processor.vocab_size
        else:
            output_dim = [lp.vocab_size for lp in self.label_processor]

        config = self.hyper_parameters
        embed_model = self.embedding.embed_model
        # Define layers for BiLSTM
        layer_stack = [
            L.Bidirectional(L.LSTM(**config['layer_bilstm1'])),
            L.Dropout(rate=0.1),
            # L.Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu'),
            # L.GlobalMaxPool1D()
        ]

        # tensor flow in Layers {tensor:=layer(tensor)}
        tensor = embed_model.output
        # tensor = L.Concatenate(axis=-1)([tensor, features_tensor])
        for layer in layer_stack:
            tensor = layer(tensor)

        tensor_non_features = L.GlobalMaxPool1D()(tensor)
        tensor_non_features_att = L.GlobalMaxPool1D()(L.Attention()([tensor, tensor]))
        tensor_non_features = L.Concatenate(axis=-1)([tensor_non_features, tensor_non_features_att])

        tensor = L.Concatenate(axis=-1)([tensor, features_tensor])
        tensor = L.Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu')(tensor)

        query_value_attention_seq = L.MultiHeadAttention(
            num_heads=2, key_dim=2, dropout=0.5
        )(tensor, tensor)

        query_encoding = L.GlobalMaxPool1D()(tensor)
        query_value_attention = L.GlobalMaxPool1D()(query_value_attention_seq)
        input_tensor = L.Concatenate()([query_encoding, query_value_attention])

        # output tensor
        input_tensor = L.Dropout(rate=0.1)(input_tensor)
        tensor_non_features = L.Dropout(rate=0.1)(tensor_non_features)
        if self.task_num == 1:
            output_tensor = L.Dense(output_dim,
                                    activation='softmax', name="output0", kernel_regularizer=l2_reg)(input_tensor)
            self.tf_model = tf.keras.Model(inputs=[embed_model.inputs, features], outputs=output_tensor)
        else:
            activations = ['softmax', 'sigmoid', 'softmax']
            inputs = [input_tensor, tensor_non_features, input_tensor]

            output_tensor = [L.Dense(output_dim[i], activation=activations[i], name="output" + str(i),
                                     kernel_regularizer=l2_reg)(inputs[i])
                             for i in range(self.task_num)]

            # output_tensor = [L.Dense(output_dim[0],
            #                          activation='softmax', name="output0", kernel_regularizer=l2_reg)(input_tensor),
            #                  L.Dense(output_dim[1],
            #                          activation='sigmoid', name="output1", kernel_regularizer=l2_reg)(input_tensor)]

            # Init model
            self.tf_model = tf.keras.Model(inputs=[embed_model.inputs, features], outputs=output_tensor)

        # plot_model(self.tf_model, to_file="D:/PycProject/TripleC/reference/model.png")
