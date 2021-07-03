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
                'units': 256,
                'return_sequences': True
            },
            'layer_dropout': {
                'rate': 0.1,
                'name': 'layer_dropout'
            },
            'layer_dropout_output': {
                'rate': 0.1,
                'name': 'layer_dropout_output'
            },
            'layer_time_distributed': {},
            'conv_layer1': {
                'filters': 128,
                'kernel_size': 4,
                'padding': 'valid',
                'activation': 'relu'
            },
            'layer_output1': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        """
        build model architectural

        BiLSTM + Convolution + Attention
        """
        features = tf.keras.Input(shape=(None, self.feature_D), name="features")
        if self.task_num == 1:
            output_dim = self.label_processor.vocab_size
        else:
            output_dims = [lp.vocab_size for lp in self.label_processor]
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model
        # Define layers for BiLSTM
        layer_stack = [
            L.Bidirectional(L.LSTM(**config['layer_bilstm1'])),
            # L.Conv1D(**config['conv_layer1']),
            L.Dropout(**config['layer_dropout'])
        ]

        # tensor flow in Layers {tensor:=layer(tensor)}
        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        # extend features
        tensor = L.Concatenate(axis=-1)([features, tensor])
        tensor = L.Conv1D(**config['conv_layer1'])(tensor)

        '''
        define attention layer
        as a nlp-rookie im wondering whether this is a right way XD
        '''
        # query_value_attention_seq = L.Attention()([tensor, tensor])
        query_value_attention_seq = L.MultiHeadAttention(
            num_heads=2, key_dim=2, dropout=0.5
        )(tensor, features)

        query_encoding = L.GlobalMaxPool1D()(tensor)
        query_value_attention = L.GlobalMaxPool1D()(query_value_attention_seq)

        input_layer = L.Concatenate(axis=-1)([query_encoding, query_value_attention])

        # output tensor
        input_layer = L.Dropout(**config['layer_dropout_output'])(input_layer)
        if self.task_num == 1:
            output_tensor = L.Dense(output_dim, activation='softmax', name="output0")(input_layer)
            self.tf_model = tf.keras.Model(inputs=[embed_model.inputs, features], outputs=output_tensor)
        else:
            # output_tensor = [L.Dense(output_dims[i], activation='sigmoid', name="output" + str(i))(input_layer)
            #                  for i in range(self.task_num)]
            output_tensor = [L.Dense(output_dims[0], activation='softmax', name="output0")(input_layer),
                             L.Dense(output_dims[1], activation='sigmoid', name="output1")(input_layer)]

            # use this activation layer as final activation to support multi-label classification
            # tensor = self._activation_layer()(tensor)

            # Init model
            self.tf_model = tf.keras.Model(inputs=[embed_model.inputs, features], outputs=output_tensor)

        # plot_model(self.tf_model, to_file="D:/PycProject/TripleC/reference/model.png")
