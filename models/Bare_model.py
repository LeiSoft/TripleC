from typing import Dict, Any

from tensorflow import keras

from kashgari.layers import L
from kashgari.tasks.classification.abc_model import ABCClassificationModel


class Bare_Model(ABCClassificationModel):
    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_output': {

            }
        }

    def build_model_arc(self) -> None:
        output_dim = self.label_processor.vocab_size

        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # build model structure in sequent way
        layer_stack = [
            L.Dense(output_dim, activation='linear', name="output0")
        ]

        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        self.tf_model: keras.Model = keras.Model(embed_model.inputs, tensor)
