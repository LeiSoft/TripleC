from utils import *
from typing import Union, List
import en_core_web_lg
import networkx as nx
from tqdm import tqdm
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Extractor:
    def __init__(self, label: bool):
        self.parser = en_core_web_lg.load()
        self.label = label

    def _build_pos_seq(self, context):

        pos_seq = []
        pos_type = set()
        context = [" ".join(t) for t in context]
        for doc in tqdm(self.parser.pipe(context)):
            text_pos = []
            for token in doc:
                pos_type.add(token.pos_)
                text_pos.append(token.pos_)
            pos_seq.append(
                " ".join(text_pos)
            )
        tokenizer = Tokenizer(num_words=len(pos_type), oov_token="<OOV>")
        tokenizer.fit_on_texts(pos_seq)
        seq = pad_sequences(
                    tokenizer.texts_to_sequences(pos_seq),
                    padding="post", truncating="post"
                )

        return seq

    def _build_graph(self, text):
        doc = self.parser(text)

        edges = []
        for token in doc:
            edges.append(('{}_{}'.format(token.head.lemma_, token.head.i),
                          '{}_{}'.format(token.lemma_, token.i)))
            for child in token.children:
                edges.append(('{}_{}'.format(token.lemma_, token.i),
                              '{}_{}'.format(child.lemma_, child.i)))
        graph = nx.Graph(edges)
        return graph, doc

    @staticmethod
    def _get_3c(task, unique_ids):
        # i dont want to pass parameters @_@
        type_ = "train"
        if unique_ids[0][2] == "T":
            type_ = "test"
        data = pd.read_csv("./datasets/" + task + "/SDP_" + type_ + ".csv", sep=',', header=0)

        title_dic, author_dic = pickle.load(open("./datasets/3c_feature_dic_"+type_+".pkl", 'rb'))
        features_3c = []
        for unique_id in unique_ids:
            num = int(unique_id[-1])
            features_3c.append([
                title_dic[data.loc[num, 'citing_title']],
                author_dic[data.loc[num, 'citing_author']]
            ])
        return features_3c

    @staticmethod
    def _tfidf(sentences):
        tk = tf.keras.preprocessing.text.Tokenizer()
        tk.fit_on_texts(sentences)
        return tk.sequences_to_matrix(tk.texts_to_sequences(sentences), mode='tfidf')

    def build_features(self, inputs: Union[str, List], **config):
        """
        return features tensor, shape: (None,None,None)
        """
        features_info = []
        if isinstance(inputs, str):
            if self.label:
                context = load_data(inputs)[0]
            else:
                context = load_non_label_data(inputs)
        else:
            context = [items[1] for items in inputs]
            # features_info = self._get_3c(config["task"], [items[0] for items in inputs])

        pos_seq = self._build_pos_seq(context)
        tfidf_matrix = self._tfidf(context)

        features = []
        # sentence id, word id
        for sid, sentence in enumerate(context):
            seq_features = []
            for wid, word in enumerate(context[sid]):
                seq_features.append(
                    [float(pos_seq[sid][wid]), float(tfidf_matrix[sid][wid])]
                )

            features.append(seq_features)
        # print(features)
        # exit(99)
        return features






