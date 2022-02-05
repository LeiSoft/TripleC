from typing import Dict, Any

from tensorflow import keras
import tensorflow as tf

from kashgari.layers import L
from kashgari.tasks.classification.abc_model import ABCClassificationModel


class Bare_Model(ABCClassificationModel):
    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_lstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_output': {

            }
        }

    def build_model_arc(self) -> None:
        l2_reg = tf.keras.regularizers.L2(0.01)
        output_dim = self.label_processor.vocab_size

        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # build model structure in sequent way
        layer_stack = [
            L.Bidirectional(L.LSTM(**config['layer_bi_lstm'])),
            L.Dropout(rate=0.1),
        ]

        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        query_encoding = L.GlobalMaxPool1D()(tensor)
        query_value_attention = L.GlobalMaxPool1D()(L.Attention()([tensor, tensor]))
        input_tensor = L.Concatenate()([query_encoding, query_value_attention])
        input_tensor = L.Dropout(rate=0.1)(input_tensor)

        output_tensor = L.Dense(output_dim,
                                activation='sigmoid', name="output0", kernel_regularizer=l2_reg)(input_tensor)
        self.tf_model: keras.Model = keras.Model(embed_model.inputs, output_tensor)
