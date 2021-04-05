from typing import Dict, Any

from tensorflow import keras
from tensorflow.python.keras import backend as K

from kashgari.tasks.classification.abc_model import ABCClassificationModel
from kashgari.layers import L

import logging

logging.basicConfig(level='DEBUG')


class SelfAtt(ABCClassificationModel):

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
            'layer_dropout': {
                'rate': 0.5,
                'name': 'layer_dropout'
            },
            'layer_time_distributed': {},
            'layer_output': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural

        RNN + Attention
        """
        output_dim = self.label_processor.vocab_size
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # Define layers for BiLSTM
        layer_stack = [
            L.Dropout(**config['layer_dropout'])
        ]

        # tensor flow in BiLSTM {tensor:=layer(tensor)}
        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        # define attention layer
        query_value_attention_seq = L.Attention()([tensor, tensor])

        query_encoding = L.GlobalMaxPool1D()(tensor)
        query_value_attention = L.GlobalMaxPool1D()(query_value_attention_seq)

        input_layer = L.Concatenate(axis=-1)([query_encoding, query_value_attention])

        # output tensor
        tensor = L.Dense(output_dim, **config['layer_output'])(input_layer)

        # Init model
        self.tf_model = keras.Model(embed_model.inputs, tensor)


class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]
