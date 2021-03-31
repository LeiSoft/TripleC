# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint.

根据pytorch-transformers.convert_pytorch_checkpoint_to_tf修改成适配tensorflow2.*的版本

修改了76行的表达，解决bert4keras无法加载模型的问题
修改前：return 'bert/{}'.format(name)
修改后：return '{}'.format(name)
Edit Credit：@FoVNull
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from pytorch_transformers import BertModel


def convert_pytorch_checkpoint_to_tf2(model: BertModel, ckpt_dir: str, model_name: str):
    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return:

    Currently supported HF models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering
    """

    tensors_to_transpose = (
        "dense.weight",
        "attention.self.query",
        "attention.self.key",
        "attention.self.value"
    )

    var_map = (
        ('layer.', 'layer_'),
        ('word_embeddings.weight', 'word_embeddings'),
        ('position_embeddings.weight', 'position_embeddings'),
        ('token_type_embeddings.weight', 'token_type_embeddings'),
        ('.', '/'),
        ('LayerNorm/weight', 'LayerNorm/gamma'),
        ('LayerNorm/bias', 'LayerNorm/beta'),
        ('weight', 'kernel')
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = model.state_dict()

    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return '{}'.format(name)

    def create_tf_var(tensor: np.ndarray, name: str, session: tf.compat.v1.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.compat.v1.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name,
                                           initializer=tf.zeros_initializer())
        session.run(tf.compat.v1.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    ops.reset_default_graph()
    with tf.compat.v1.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].cpu().numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))
