import math
import os
from typing import List, Optional, Union
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from kashgari.embeddings import WordEmbedding, TransformerEmbedding
from kashgari_local import XLNetEmbedding, MPNetEmbedding, TransformerEmbeddingPE
from kashgari.tokenizers import BertTokenizer
from kashgari.tasks.classification import BiLSTM_Model
from sklearn.model_selection import train_test_split
from transformers import XLNetTokenizer, MPNetTokenizer
from transformers import BertTokenizer as HFBertTokenizer

from models.RCNN_Att import RCNN_Att_Model
from models.scaffold_model import Scaffold_Model
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
        self.extractor = Extractor()

    def _embedding(self, **params):
        if self.model_type == "w2v":
            embedding = WordEmbedding(self.vector_path)
        elif self.model_type == "xlnet":
            # 输入语料用于构建xlnet词表，生成器形式
            embedding = XLNetEmbedding(self.model_folder, _xlnet_corpus_gen(params["xlnet_corpus"]))
        elif self.model_type == "mpnet":
            embedding = MPNetEmbedding(self.model_folder)
        else:
            embedding = TransformerEmbeddingPE(self.vocab_path, self.config_path, self.checkpoint_path,
                                               model_type=self.model_type)

        return embedding

    def _tokenizing(self, _x):
        print('###### tokenizing... #########')
        sentences_tokenized = _x

        if self.model_type == 'bert':
            tokenizer = BertTokenizer.load_from_vocab_file(self.vocab_path)
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in tqdm(_x)]
        if self.model_type == 'xlnet':
            tokenizer = XLNetTokenizer(self.model_folder + "/spiece.model", )
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in tqdm(_x)]
        if self.model_type == 'mpnet':
            tokenizer = MPNetTokenizer(self.model_folder + '/vocab.txt')
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in tqdm(_x)]
        if self.model_type == 'bert_hf':
            tokenizer = HFBertTokenizer(self.model_folder + '/vocab.txt')
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in tqdm(_x)]

        return sentences_tokenized

    # 3c-shared task评测用，不适用于后续实验
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

    # 适用单标签任务/多标签任务+特征融合
    def train_sci_cross(self, path, **params):
        x_data, y_data = load_data(path + "all.tsv")
        
        # 3c数据集错例分析用
        _, output_vx, _, output_vy = train_test_split(
            x_data, y_data, test_size=1000, shuffle=True, random_state=2)
        output_x, _, output_y, _= train_test_split(
            output_vx, output_vy, test_size=500, shuffle=True, random_state=2)
        output_x = [' '.join(i) for i in output_x]

        if params['task_num'] > 1:
            if self.task_type == 'scicite':
                x_data = self._tokenizing(x_data)
                y_data = get_multi_label(path + "all.jsonl", y_data, self.task_type)
            if self.task_type == '3c':
                x_data = self._tokenizing(x_data)
                y_data = get_multi_label('./datasets/3c-shared/influence/train.tsv', y_data, self.task_type)
        if self.task_type == '3c':
            all_features = self.extractor.build_features(path + "all.tsv", task_type=self.task_type)
        else:
            all_features = self.extractor.build_features(path + "all.jsonl", task_type=self.task_type)

        data_gen = None
        assert params['cross'] in ['fold', 'random', 'repeat', "solid_rate"], 'cross validation must be fold/random/repeat/solid_rate'
        if params['cross'] == 'fold':
            data_gen = self.k_fold(params['fold'], x_data, y_data, all_features, params['v_size'])
        if params['cross'] == 'random':
            data_gen = self.random_cross(params['fold'], x_data, y_data, all_features,
                                         params['t_v_size'], params['v_size'])
        # acl-arc的固定数据集重复实验
        if params['cross'] == 'repeat':
            x_train, y_train = load_data(path + 'train.tsv')
            x_vali, y_vali = load_data(path + 'dev.tsv')
            x_test, y_test = load_data(path + 'test.tsv')
            f_train = self.extractor.build_features(path + "train.jsonl", task_type=self.task_type)
            f_vali = self.extractor.build_features(path + "dev.jsonl", task_type=self.task_type)
            f_test = self.extractor.build_features(path + "test.jsonl", task_type=self.task_type)
            data_gen = self.repeat_val(params['fold'], x_train, y_train, f_train, x_vali, y_vali, f_vali, x_test, y_test, f_test)
        
        if params['cross'] == 'solid_rate':
            data_gen = self.solid_rate(params['fold'], params['test_rate'], x_data, y_data, all_features, params['v_size'])

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]
        reports = []
        sample_saving_flag = 100
        for x_test, y_test, test_features, \
            x_vali, y_vali, vali_features, \
            x_train, y_train, train_features, seed in data_gen:

            if self.model_type == 'xlnet':
                embedding = self._embedding(xlnet_corpus=path + "all.tsv")
            else:
                embedding = self._embedding()

            print("train-{}, vali-{}, test-{}".format(len(x_train), len(x_vali), len(x_test)))
            if params['with_feature']:
                model = RCNN_Att_Model(embedding, feature_D=len(train_features[0][0]), task_num=params['task_num'])
                model.fit(x_train=(x_train, train_features), y_train=y_train,
                          x_validate=(x_vali, vali_features), y_validate=y_vali,
                          batch_size=32, epochs=1, callbacks=None)
                rep, main_pred = self.evaluate(model, x_test, y_test, test_features)

                # 错例 保存，通过sample_saving_flag控制是否执行，非3c数据集需要对output_x进行相应改动
                if sample_saving_flag < rep[0]['detail']['macro avg']['f1-score']:
                    with open('./reference/samples_result_3c.tsv', 'w', encoding='utf-8') as f:
                        f.write('text\tpred\ttrue\n')
                        for i in range(len(main_pred)):
                            f.write(f'{output_x[i]}\t{main_pred[i]}\t{output_y[i]}\n')
                    sample_saving_flag = rep[0]['detail']['macro avg']['f1-score']
                print(sample_saving_flag, '#'*50)
            else:
                model = Bare_Model(embedding)
                model.fit(x_train, y_train, x_vali, y_vali, batch_size=32, epochs=15)
                rep = model.evaluate(x_test, y_test, batch_size=32)

            reports.append(rep)

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
                f.write(res+f'\nstd:{np.std(f1)}\n')
            print(f'std:{np.std(f1)}')
    
    # 参照scaffold论文，支持非多标签形式、输入数据不同的三任务结构
    def train_scaffold(self, path, data_split_type):
        worth, sections = get_scaffolds(path)
        worth_text = self._tokenizing([text for text, _ in worth])
        sections_text = self._tokenizing([text for text, _ in sections])
        worth_label = [label for _, label in worth]
        sections_label = [label for _, label in sections]

        wt_train, wt_test, wl_train, wl_test = train_test_split(
            worth_text, worth_label, test_size=0.15, random_state=114
        )
        st_train, st_test, sl_train, sl_test = train_test_split(
            sections_text, sections_label, test_size=0.15, random_state=810
        )

        if data_split_type == 'random':
            x_data, y_data = load_data(path + "all.tsv")

            all_features = self.extractor.build_features(path + "all.jsonl", task_type=self.task_type)
            x_data = self._tokenizing(x_data)
            x_train, x_test, y_train, y_test, f_train, f_test = train_test_split(
                x_data, y_data, all_features, test_size=139, random_state=2
            )

        if data_split_type == 'solid':
            x_train, y_train = load_data(path + 'train.tsv')
            x_vali, y_vali = load_data(path + 'dev.tsv')
            x_train += x_vali
            y_train += y_vali
            x_train = self._tokenizing(x_train)
            x_test, y_test = load_data(path + 'test.tsv')
            x_test = self._tokenizing(x_test)
            f_train = self.extractor.build_features(path + "train.jsonl", task_type=self.task_type) + \
                self.extractor.build_features(path + "dev.jsonl", task_type=self.task_type)
            f_test = self.extractor.build_features(path + "test.jsonl", task_type=self.task_type)
        
        if data_split_type == 'solid_rate':
            x_data, y_data = load_data(path + "all.tsv")
            all_features = self.extractor.build_features(path + "all.jsonl", task_type=self.task_type)

            # 通过solid_seed控制随机，然后按照比例抽取测试集。
            # 保证了测试集的比例固定，同时可以增加随机性
            solid_seed = 42
            np.random.seed(solid_seed)
            np.random.shuffle(x_data)
            np.random.seed(solid_seed)
            np.random.shuffle(y_data)
            np.random.seed(solid_seed)
            np.random.shuffle(all_features)

            x_data = self._tokenizing(x_data)

            test_rate = [71, 25, 5, 5, 7, 26]
            x_train, y_train, x_test, y_test, f_train, f_test = [], [], [], [], [], []
            for i in range(len(x_data)):
                label = int(y_data[i]) 
                if test_rate[label] > 0:
                    x_test.append(x_data[i])
                    y_test.append(y_data[i])
                    f_test.append(all_features[i])
                    test_rate[label] -= 1
                else:
                    x_train.append(x_data[i])
                    y_train.append(y_data[i])
                    f_train.append(all_features[i])

        output_dims = [len(set(y_test)), len(set(worth_label)), len(set(sections_label))]
        reports = []
        fold_num = 1
        output_x, output_y = [' '.join(_) for _ in x_test], y_test
        for i in range(fold_num):
            print(f'################{i+1}/{fold_num}#################')

            model = Scaffold_Model(self.model_folder, self.model_type, output_dims, feature_D=len(f_train[0][0]))
            model.fit(x_train, y_train, wt_train, wl_train, st_train, sl_train, f_train,
                    batch_size=32, epoch=15)

            report = model.evaluate(x_test, y_test, wt_test, wl_test, st_test, sl_test, f_test,
                                    batch_size=32, output_x_y=(output_x, output_y))
            
            reports.append(report)

        with open(path + "scaffold_reports.tsv", 'w', encoding='utf-8') as f:
            for i in range(len(reports[0])):
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
                    .format(i, fold_num,
                            sum(acc) / fold_num, sum(p) / fold_num, sum(r) / fold_num, sum(f1) / fold_num,
                            max(acc), max(p), max(r), max(f1), min(acc), min(p), min(r), min(f1))
                f.write(res+f'\nstd:{np.std(f1)}\n')
            print(f'std:{np.std(f1)}')
        

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
    def k_fold(k, x, y, f, part):
        np.random.seed(6)
        np.random.shuffle(x)
        np.random.shuffle(y)
        np.random.shuffle(f)
        step = len(x) // k
        for i in range(0, len(x), step):
            yield x[i:i + part], y[i:i + part], f[i:i + part], \
                  x[i + part:i + part * 2], y[i + part:i + part * 2], f[i + part:i + part * 2], \
                  x[:i] + x[i + part * 2:], y[:i] + y[i + part * 2:], f[:i] + f[i + part * 2:], -1

    @staticmethod
    def random_cross(k, x, y, f, t_v_size, v_size):
        # 3c-intent 2 acl-arc 6/82
        seeds = [2 for _ in range(k)]
        for idx, seed in enumerate(seeds):
            print(f'################{idx+1}/{k}#################')
            x_train, x_vali, y_train, y_vali, f_train, f_vali = train_test_split(
                x, y, f, test_size=t_v_size, shuffle=True, random_state=seed)
            x_test, x_vali, y_test, y_vali, f_test, f_vali = train_test_split(
                x_vali, y_vali, f_vali, test_size=v_size, shuffle=True, random_state=seed)

            yield x_test, y_test, f_test, x_vali, y_vali, f_vali, x_train, y_train, f_train, seed
    
    @staticmethod
    def solid_rate(k, test_rate, x, y, f, v_size):
        seeds = [42 for _ in range(k)]
        for idx, solid_seed in enumerate(seeds):
            print(f'################{idx+1}/{k}#################')
            np.random.seed(solid_seed)
            np.random.shuffle(x)
            np.random.seed(solid_seed)
            np.random.shuffle(y)
            np.random.seed(solid_seed)
            np.random.shuffle(f)

            x_train, y_train, x_test, y_test, f_train, f_test = [], [], [], [], [], []
            for i in range(len(x)):
                label = int(y[i]) if isinstance(y[i], str) else int(y[i][0])
                if test_rate[label] > 0:
                    x_test.append(x[i])
                    y_test.append(y[i])
                    f_test.append(f[i])
                    test_rate[label] -= 1
                else:
                    x_train.append(x[i])
                    y_train.append(y[i])
                    f_train.append(f[i])
            x_train, x_vali, y_train, y_vali, f_train, f_vali = train_test_split(
                x, y, f, test_size=v_size, shuffle=True, random_state=solid_seed)
            
            yield x_test, y_test, f_test, x_vali, y_vali, f_vali, x_train, y_train, f_train, solid_seed



    @staticmethod
    def repeat_val(k, x_train, y_train, f_train, x_vali, y_vali, f_vali, x_test, y_test, f_test):
        for _ in range(k):
            yield x_test, y_test, f_test, x_vali, y_vali, f_vali, x_train, y_train, f_train, -1
