from kashgari.embeddings import TransformerEmbedding, BertEmbedding
from kashgari.processors import SequenceProcessor
from kashgari.embeddings import WordEmbedding
from kashgari.tokenizers import BertTokenizer
from kashgari.tasks.classification import CNN_Attention_Model, BiLSTM_Model
from models.BiLSTM_Conv_Att import BiLSTM_Conv_Att_Model
import json

from sklearn.model_selection import train_test_split
from utils import *


class Trainer:
    def __init__(self, _args):
        self.model_type = _args.model_type
        if _args.model_type == "w2v":
            self.vocab_path = _args.model_folder+"/vocab.txt"
            self.vector_path = _args.model_folder+"/3C.vec"
        else:
            self.checkpoint_path = _args.model_folder + '/bert_model.ckpt'
            self.config_path = _args.model_folder + '/bert_config.json'
            self.vocab_path = _args.model_folder + '/vocab.txt'
        self.model_folder = _args.model_folder
        self.task_type = _args.task_type

    def _embedding(self):
        if self.model_type == "w2v":
            embedding = WordEmbedding(self.vector_path)
        else:
            embedding = TransformerEmbedding(self.vocab_path, self.config_path,  self.checkpoint_path,
                                             model_type=self.model_type)

        return embedding

    def _tokenizing(self, _x):
        sentences_tokenized = _x

        if self.model_type != 'w2v':
            tokenizer = BertTokenizer.load_from_vocab_file(self.vocab_path)
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in _x]

        return sentences_tokenized

    def train(self, path, **params):
        x_data, y_data = load_data(path)

        embedding = self._embedding()
        x_data = self._tokenizing(x_data)

        if params["test"]:
            x_train, x_test, y_train, y_test = train_test_split(
                x_data, y_data, test_size=params["test_size"], random_state=3
            )
        else:
            x_train, y_train = x_data, y_data
        model = BiLSTM_Conv_Att_Model(embedding)

        if params["validation"]:
            x_train, x_vali, y_train, y_vali = train_test_split(
                x_train, y_train, test_size=params["test_size"]
            )
            model.fit(x_train, y_train, x_vali, y_vali, batch_size=64, epochs=20, callbacks=None, fit_kwargs=None)
        else:
            model.fit(x_train, y_train, batch_size=64, epochs=20, callbacks=None, fit_kwargs=None)

        if params["test"]:
            self.evaluate(model, x_test, y_test)

        x_interface = load_test_data("./datasets/"+self.task_type+"/test.tsv")
        y_interface = model.predict(x_interface, batch_size=64, truncating=True, predict_kwargs=None)
        self._generate(y_interface)

        return model

    @staticmethod
    def evaluate(model, x_test, y_test):
        report = model.evaluate(x_test, y_test, batch_size=64, digits=4, truncating=False)
        print(report)
        # model.save("./models/output")

    def _generate(self, y):
        if self.task_type == "intent":
            head = 'unique_id,citation_class_label'
        else:
            head = 'unique_id,citation_influence_label'

        with open("./datasets/"+self.task_type+"_prediction.csv", 'w', encoding='utf-8') as f:
            f.write(head+"\n")
            for index, label in enumerate(y):
                f.write("CCT"+str(index+1)+","+label+"\n")
