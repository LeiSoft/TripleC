import math
import os
from typing import List, Optional, Union

import random
from kashgari.embeddings import TransformerEmbedding, WordEmbedding
from kashgari_local import XLNetEmbedding, MPNetEmbedding
from kashgari.tokenizers import BertTokenizer
from kashgari.tasks.classification import BiLSTM_Model
from sklearn.model_selection import train_test_split
from transformers import XLNetTokenizer, MPNetTokenizer

from models.RCNN_Att import RCNN_Att_Model
from models.Bare_model import Bare_Model
from features.extractor import Extractor
from utils import *


def _xlnet_corpus_gen(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            yield line.strip().split("\t")[1]


class Trainer:
    def __init__(self, _args):
        self.model_type = _args.model_type
        if _args.model_type == "w2v":
            self.vocab_path = _args.model_folder + "/vocab.txt"
            self.vector_path = _args.model_folder + "/3C.vec"
        if _args.model_type == "bert":
            self.checkpoint_path = _args.model_folder + '/bert_model.ckpt'
            self.config_path = _args.model_folder + '/bert_config.json'
            self.vocab_path = _args.model_folder + '/vocab.txt'
        self.model_folder = _args.model_folder
        self.task_type = _args.task_type
        self.extractor = Extractor(True)

    def _embedding(self, **params):
        if self.model_type == "w2v":
            embedding = WordEmbedding(self.vector_path)
        elif self.model_type == "xlnet":
            # 输入语料用于构建xlnet词表，生成器形式
            embedding = XLNetEmbedding(self.model_folder, _xlnet_corpus_gen(params["xlnet_corpus"]))
        elif self.model_type == "mpnet":
            embedding = MPNetEmbedding(self.model_folder)
        else:
            embedding = TransformerEmbedding(self.vocab_path, self.config_path, self.checkpoint_path,
                                             model_type=self.model_type)

        return embedding

    def _tokenizing(self, _x):
        sentences_tokenized = _x

        if self.model_type == 'bert':
            tokenizer = BertTokenizer.load_from_vocab_file(self.vocab_path)
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in _x]
        if self.model_type == 'xlnet':
            tokenizer = XLNetTokenizer(self.model_folder + "/spiece.model", )
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in _x]
        if self.model_type == 'mpnet':
            tokenizer = MPNetTokenizer(self.model_folder + '/vocab.txt')
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in _x]

        return sentences_tokenized

    def train_3c(self, path, **params):
        x_data, y_data = load_data(path)
        # pointless, just for ignoring the warning
        x_test, y_test = None, None

        if params['task_num'] > 1:
            assert isinstance(params['multi_paths'], List), "multi-task needs multi corpus in 3C@SDP2021"
            y_data = add_label(y_data, params['multi_paths'])

        if self.model_type == 'xlnet':
            embedding = self._embedding(xlnet_corpus=path + "all.tsv")
        elif self.model_type == 'mpnet':
            embedding = self._embedding()
        else:
            embedding = self._embedding()

        model = RCNN_Att_Model(embedding, task_num=params['task_num'], feature_D=21)
        x_data = self._tokenizing(x_data)

        if params["test_size"]:
            x_train, x_test, y_train, y_test = train_test_split(
                x_data, y_data, test_size=params["test_size"], random_state=810
            )
        else:
            x_train, y_train = x_data, y_data

        if params["vali_size"]:
            x_train, x_vali, y_train, y_vali = train_test_split(
                x_train, y_train, test_size=params["vali_size"], random_state=514
            )
            train_features = self.extractor.build_features(x_train, task=self.task_type)
            vali_features = self.extractor.build_features(x_vali, task=self.task_type)

            model.fit((x_train, train_features), y_train,
                      (x_vali, vali_features), y_vali,
                      batch_size=32, epochs=15, callbacks=None, fit_kwargs=None)
        else:
            train_features = self.extractor.build_features(x_train, task=self.task_type)

            model.fit((x_train, train_features), y_train,
                      batch_size=32, epochs=10, callbacks=None, fit_kwargs=None)

        if params["test_size"]:
            assert x_test and y_test, "unexpectable error! test data shouldn't be None, check it out"
            test_features = self.extractor.build_features(x_test, task=self.task_type)

            self.evaluate(model, x_test, y_test, test_features)

        x_interface = load_non_label_data("./datasets/" + self.task_type + "/test.tsv")
        x_interface_pure = [items[1] for items in x_interface]
        interface_features = self.extractor.build_features(x_interface_pure, task=self.task_type)
        y_interface = model.predict((x_interface_pure, interface_features),
                                    batch_size=32, truncating=True, predict_kwargs=None)
        self._generate(y_interface[0])
        os.system('kaggle competitions submit -c 3c-shared-task-purpose-v2 '
                  '-f /home/sz/project/TripleC/datasets/3c-shared/intent_prediction.csv -m "Message"')

        return model

    def train_scicite(self, path, **params):
        x_train, y_train = load_data(path + "train.tsv")
        x_vali, y_vali = load_data(path + "dev.tsv")
        x_test, y_test = load_data(path + "test.tsv")

        if self.model_type == 'xlnet':
            embedding = self._embedding(xlnet_corpus=path + "all.tsv")
            x_train = self._tokenizing(x_train)
            x_vali = self._tokenizing(x_vali)
            x_test = self._tokenizing(x_test)
        elif self.model_type == 'mpnet':
            embedding = self._embedding()
            x_train = self._tokenizing(x_train)
            x_vali = self._tokenizing(x_vali)
            x_test = self._tokenizing(x_test)
        else:
            embedding = self._embedding()

        if params['task_num'] > 1:
            y_train = get_multi_label(path + "train.jsonl", y_train)
            y_vali = get_multi_label(path + "dev.jsonl", y_vali)
            y_test = get_multi_label(path + "test.jsonl", y_test)
        train_features = self.extractor.build_features(path + "train.tsv")
        vali_features = self.extractor.build_features(path + "dev.tsv")

        model = RCNN_Att_Model(embedding, feature_D=len(train_features[0][0]), task_num=params['task_num'])
        # model = BiLSTM_Model(embedding)

        model.fit(x_train=(x_train, train_features), y_train=y_train,
                  x_validate=(x_vali, vali_features), y_validate=y_vali,
                  batch_size=32, epochs=15, callbacks=None, fit_kwargs=None)
        # model.fit(x_train, y_train, x_vali, y_vali, epochs=15, batch_size=32)
        # return model.evaluate(x_test, y_test, batch_size=32, digits=4, truncating=False)

        test_features = self.extractor.build_features(path + "test.tsv")
        return self.evaluate(model, x_test, y_test, test_features)

    # 支持交叉验证
    def train_scicite_cross(self, path, **params):
        x_data, y_data = load_data(path + "all.tsv")
        if self.model_type == 'xlnet':
            embedding = self._embedding(xlnet_corpus=path + "all.tsv")
            x_data = self._tokenizing(x_data)
        elif self.model_type == 'mpnet':
            embedding = self._embedding()
            x_data = self._tokenizing(x_data)
        else:
            embedding = self._embedding()

        if params['task_num'] > 1:
            if self.task_type == 'scicite':
                y_data = get_multi_label(path + "all.jsonl", y_data)
            if self.task_type == 'acl-arc':
                y_data = get_multi_label(path + "/scaffolds/cite-worthiness-scaffold-train.jsonl", y_data)
        all_features = self.extractor.build_features(path + "all.tsv")

        data_gen = None
        assert params['cross'] in ['fold', 'random'], 'cross validation must be fold or random'
        if params['cross'] == 'fold':
            data_gen = self.k_fold(params['fold'], x_data, y_data, all_features)
        if params['cross'] == 'random':
            data_gen = self.random_cross(params['fold'], x_data, y_data, all_features,
                                         params['t_v_size'], params['v_size'])

        reports = []
        for x_test, y_test, test_features, \
            x_vali, y_vali, vali_features, \
            x_train, y_train, train_features in data_gen:
            model = RCNN_Att_Model(embedding, feature_D=len(train_features[0][0]), task_num=params['task_num'])
            print("train-{}, vali-{}, test-{}".format(len(x_train), len(x_vali), len(x_test)))

            model.fit(x_train=(x_train, train_features), y_train=y_train,
                      x_validate=(x_vali, vali_features), y_validate=y_vali,
                      batch_size=32, epochs=15, callbacks=None, fit_kwargs=None)

            reports.append(self.evaluate(model, x_test, y_test, test_features))

        with open(path + "reports.tsv", 'w', encoding='utf-8') as f:
            for i in range(params["task_num"]):
                acc, p, r, f1 = [], [], [], []
                for report in reports:
                    acc.append(report[i]['detail']['accuracy'])
                    p.append(report[i]['detail']['macro avg']['precision'])
                    r.append(report[i]['detail']['macro avg']['recall'])
                    f1.append(report[i]['detail']['macro avg']['f1-score'])
                res = "task{};{}-Fold\n" \
                      "avg-macro:acc={}; p={}; r={}; f1={}\n" \
                      "max-macro:acc={}; p={}; r={}; f1={}\n" \
                      "min-macro:acc={}; p={}; r={}; f1={}\n" \
                    .format(i, params['fold'],
                            sum(acc) / params['fold'], sum(p) / params['fold'], sum(r) / params['fold'], sum(f1) / params['fold'],
                            max(acc), max(p), max(r), max(f1), min(acc), min(p), min(r), min(f1))
                f.write(res)
                if sum(f1) / params['fold'] > 0.7 or sum(acc) / params['fold'] > 0.79:
                    print(sum(f1) / params['fold'], sum(acc) / params['fold'])
                    exit(99)

    @staticmethod
    def evaluate(model, x_test_pure, y_test, features):
        # y_test转置，便于model.evaluate处理多任务输出
        y_test = list(map(lambda x: list(x), zip(*y_test)))
        reports = model.evaluate((x_test_pure, features), y_test, batch_size=32)
        # for report in reports:
        #     print(report)
        # model.save("./models/output")
        return reports

    def _generate(self, y):
        if self.task_type == "3c-shared/intent":
            head = 'unique_id,citation_class_label'
        else:
            head = 'unique_id,citation_influence_label'

        with open("./datasets/" + self.task_type + "_prediction.csv", 'w', encoding='utf-8') as f:
            f.write(head + "\n")
            for index, label in enumerate(y):
                f.write("CCT" + str(index + 1) + "," + label + "\n")

    @staticmethod
    def k_fold(k, x, y, f):
        part = math.ceil((len(x) / (k * 2)))
        for i in range(0, len(x), part * 2):
            yield x[i:i + part], y[i:i + part], f[i:i + part], \
                  x[i + part:i + part * 2], y[i + part:i + part * 2], f[i + part:i + part * 2], \
                  x[:i] + x[i + part * 2:], y[:i] + y[i + part * 2:], f[:i] + f[i + part * 2:]

    @staticmethod
    def random_cross(k, x, y, f, t_v_size, v_size):
        seeds = [random.randint(0, 1000) for _ in range(k)]
        for seed in seeds:
            x_train, x_vali, y_train, y_vali, f_train, f_vali = train_test_split(
                x, y, f, test_size=t_v_size, shuffle=True, random_state=seed)
            x_test, x_vali, y_test, y_vali, f_test, f_vali = train_test_split(
                x_vali, y_vali, f_vali, test_size=v_size, shuffle=True, random_state=seed)

            yield x_test, y_test, f_test, x_vali, y_vali, f_vali, x_train, y_train, f_train
