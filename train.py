from typing import List, Optional, Union
from kashgari.embeddings import TransformerEmbedding, BertEmbedding
from kashgari.embeddings import WordEmbedding
from kashgari.tokenizers import BertTokenizer
from kashgari.tasks.classification import CNN_Attention_Model, BiLSTM_Model
from sklearn.model_selection import train_test_split

from models.RCNN_Att import RCNN_Att_Model
from models.SelfAtt import SelfAtt
from features.extractor import Extractor
from utils import *


class Trainer:
    def __init__(self, _args):
        self.model_type = _args.model_type
        if _args.model_type == "w2v":
            self.vocab_path = _args.model_folder + "/vocab.txt"
            self.vector_path = _args.model_folder + "/3C.vec"
        else:
            self.checkpoint_path = _args.model_folder + '/bert_model.ckpt'
            self.config_path = _args.model_folder + '/bert_config.json'
            self.vocab_path = _args.model_folder + '/vocab.txt'
        self.model_folder = _args.model_folder
        self.task_type = _args.task_type
        self.multi_paths = _args.multi_label
        self.extractor = Extractor(True)

    def _embedding(self):
        if self.model_type == "w2v":
            embedding = WordEmbedding(self.vector_path)
        else:
            embedding = TransformerEmbedding(self.vocab_path, self.config_path, self.checkpoint_path,
                                             model_type=self.model_type)

        return embedding

    def _tokenizing(self, _x):
        sentences_tokenized = _x

        if self.model_type != 'w2v':
            tokenizer = BertTokenizer.load_from_vocab_file(self.vocab_path)
            sentences_tokenized = [
                (seq[0], tokenizer.tokenize(" ".join(seq[1])))
                for seq in _x]

        return sentences_tokenized

    def train_3c(self, path, **params):
        x_data, y_data = load_data(path)
        # pointless, just for ignoring the warning
        x_test, y_test = None, None

        multi_label = False
        if isinstance(self.multi_paths, List):
            y_data = add_label(y_data, self.multi_paths)
            multi_label = True

        embedding = self._embedding()
        model = RCNN_Att_Model(embedding, multi_label=multi_label, feature_D=2)

        x_data = self._tokenizing(x_data)

        if params["test_size"]:
            x_train, x_test, y_train, y_test = train_test_split(
                x_data, y_data, test_size=params["test_size"], random_state=810
            )
        else:
            x_train, y_train = x_data, y_data

        if params["vali_size"]:
            x_train, x_vali, y_train, y_vali = train_test_split(
                x_train, y_train, test_size=params["vali_size"], random_state=893
            )
            train_features = self.extractor.build_features(x_train, task=self.task_type)
            vali_features = self.extractor.build_features(x_vali, task=self.task_type)

            x_train_pure = [items[1] for items in x_train]
            x_vali_pure = [items[1] for items in x_vali]

            model.fit((x_train_pure, train_features), y_train,
                      (x_vali_pure, vali_features), y_vali,
                      batch_size=64, epochs=15, callbacks=None, fit_kwargs=None)
        else:
            train_features = self.extractor.build_features(x_train, task=self.task_type)
            x_train_pure = [items[1] for items in x_train]

            model.fit((x_train_pure, train_features), y_train,
                      batch_size=64, epochs=25, callbacks=None, fit_kwargs=None)

        if params["test_size"]:
            assert x_test and y_test, "unexpectable error! test data shouldn't be None, check it out"
            test_features = self.extractor.build_features(x_test, task=self.task_type)

            x_test_pure = [items[1] for items in x_test]
            self.evaluate(model, x_test_pure, y_test, test_features)

        x_interface = load_non_label_data("./datasets/" + self.task_type + "/test.tsv")
        interface_features = self.extractor.build_features(x_interface, task=self.task_type)
        x_interface_pure = [items[1] for items in x_interface]
        y_interface = model.predict((x_interface_pure, interface_features),
                                    batch_size=64, truncating=True, predict_kwargs=None)
        self._generate(y_interface)

        return model

    def train_scicite(self, path, **params):
        x_train, y_train = load_data(path + "train.tsv")
        x_vali, y_vali = load_data(path + "dev.tsv")
        x_test, y_test = load_data(path + "test.tsv")

        if params['task_num'] > 1:
            y_train = get_multi_label(path + "train.jsonl", y_train)
            y_vali = get_multi_label(path + "dev.jsonl", y_vali)
            y_test = get_multi_label(path + "test.jsonl", y_test)

        embedding = self._embedding()
        model = RCNN_Att_Model(embedding, feature_D=2, task_num=params['task_num'])

        train_features = self.extractor.build_features(path + "train.tsv")
        vali_features = self.extractor.build_features(path + "dev.tsv")

        model.fit(x_train=(x_train, train_features), y_train=y_train,
                  x_validate=(x_vali, vali_features), y_validate=y_vali,
                  batch_size=64, epochs=20, callbacks=None, fit_kwargs=None)

        test_features = self.extractor.build_features(path + "test.tsv")
        self.evaluate(model, x_test, y_test, test_features)

    @staticmethod
    def evaluate(model, x_test_pure, y_test, features):
        # y_test转置，便于model.evaluate处理多任务输出
        y_test = list(map(lambda x: list(x), zip(*y_test)))
        reports = model.evaluate((x_test_pure, features), y_test, batch_size=64, digits=4, truncating=False)
        # 适配多任务
        for report in reports:
            print(report)
        # model.save("./models/output")

    def _generate(self, y):
        if self.task_type == "intent":
            head = 'unique_id,citation_class_label'
        else:
            head = 'unique_id,citation_influence_label'

        with open("./datasets/" + self.task_type + "_prediction.csv", 'w', encoding='utf-8') as f:
            f.write(head + "\n")
            for index, label in enumerate(y):
                f.write("CCT" + str(index + 1) + "," + label + "\n")
