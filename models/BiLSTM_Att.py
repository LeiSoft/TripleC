from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.classification.abc_model import ABCClassificationModel
from kashgari.layers import L

import logging
logging.basicConfig(level='DEBUG')


class BiLSTM_Att_Model(ABCClassificationModel):

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
            'layer_dropout': {
                'rate': 0.5,
                'name': 'layer_dropout'
            },
            'layer_dropout_output': {
                'rate': 0.5,
                'name': 'layer_dropout_output'
            },
            'layer_time_distributed': {},
            'layer_output': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural

        BiLSTM + Attention
        """
        output_dim = self.label_processor.vocab_size
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # Define layers for BiLSTM
        layer_stack = [
            L.Bidirectional(L.LSTM(**config['layer_bilstm1']), name='layer_bilstm1'),
            L.Dropout(**config['layer_dropout'])
        ]

        # tensor flow in BiLSTM {tensor:=layer(tensor)}
        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        '''
        define attention layer
        as a nlp-rookie im wondering whether this is a right way XD
        '''
        query_value_attention_seq = L.Attention()([tensor, tensor])

        query_encoding = L.GlobalMaxPool1D()(tensor)
        query_value_attention = L.GlobalMaxPool1D()(query_value_attention_seq)

        input_layer = L.Concatenate(axis=-1)([query_encoding, query_value_attention])

        # output tensor
        input_layer = L.Dropout(**config['layer_dropout_output'])(input_layer)
        tensor = L.Dense(output_dim, **config['layer_output'])(input_layer)

        # use this activation layer as final activation to support multi-label classification
        # tensor = self._activation_layer()(tensor)

        # Init model
        self.tf_model = keras.Model(embed_model.inputs, tensor)