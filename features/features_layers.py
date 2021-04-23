from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.utils import tf_utils

import tensorflow as tf
import numpy as np


class FeaturesFusion(Concatenate):
    def __init__(self, axis=-1, **kwargs):
        super(FeaturesFusion, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True
        self._reshape_required = False

        self.counter = 1

    def call(self, inputs):
        features = inputs[0]
        data = inputs[1]