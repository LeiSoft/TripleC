from utils import *
from typing import Union, List
import en_core_web_lg
import networkx as nx
from tqdm import tqdm
import pandas as pd
import pickle
import regex
import nltk
import string

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

        return seq, len(tokenizer.word_counts)

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

        title_dic, author_dic = pickle.load(open("./datasets/3c_feature_dic_" + type_ + ".pkl", 'rb'))
        features_3c = []
        for unique_id in unique_ids:
            num = int(unique_id[-1])
            features_3c.append([
                title_dic[data.loc[num, 'citing_title']],
                author_dic[data.loc[num, 'citing_author']]
            ])
        return features_3c

    @staticmethod
    def _get_scicite(path):
        pass

    @staticmethod
    def _tfidf(sentences):
        tk = tf.keras.preprocessing.text.Tokenizer()
        tk.fit_on_texts(sentences)
        return tk.sequences_to_matrix(tk.texts_to_sequences(sentences), mode='tfidf')

    @staticmethod
    def get_pos_structure(tokenized):
        pos_tags = nltk.pos_tag(tokenized)
        pos = []
        for pos_tag in pos_tags:
            if pos_tag[0] == 'REFERENCE':
                pos.append(pos_tag[0])
            else:
                pos.append(pos_tag[1])
        return ' '.join([tag for tag in pos if tag not in string.punctuation])

    @staticmethod
    def find_pos_patterns(pos_sentence):
        pattern_0 = regex.compile('^.*REFERENCE VB[DPZN].*$').match(pos_sentence) is not None
        pattern_1 = regex.compile('^.*VB[DPZ] VB[GN].*$').match(pos_sentence) is not None
        pattern_2 = regex.compile('^.*VB[DGPZN]? (RB[RS]? )*VBN.*$').match(pos_sentence) is not None
        pattern_3 = regex.compile('^.*MD (RB[RS]? )*VB (RB[RS]? )*VBN.*$').match(pos_sentence) is not None
        pattern_4 = regex.compile('^(RB[RS]? )*PRP (RB[RS]? )*V.*$').match(pos_sentence) is not None
        pattern_5 = regex.compile('^.*VBG (NNP )*(CC )*(NNP ).*$').match(pos_sentence) is not None
        return [pattern_0, pattern_1, pattern_2, pattern_3, pattern_4, pattern_5]

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
            context = inputs
            # features_info = self._get_3c(config["task"], [items[0] for items in inputs])

        pos_seq, pos_num = self._build_pos_seq(context)
        tfidf_matrix = self._tfidf(context)

        features = []
        # sentence id, word id
        for sid in range(len(pos_seq)):
            seq_features = []
            # for wid in range(len((pos_seq[sid]))):
            #     token_features = list(tf.keras.utils.to_categorical(
            #         pos_seq[sid][wid], num_classes=17, dtype='float32'))
            #     token_features.append(float(tfidf_matrix[sid][wid]))
            #
            #     seq_features.append(
            #         token_features
            #     )
            # features.append(seq_features)

            pos_structure = self.get_pos_structure(context[sid])
            pos_patterns = [abs(-pp) for pp in self.find_pos_patterns(pos_structure)]
            features.append([pos_patterns])

        return features
