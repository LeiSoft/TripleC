from kashgari.embeddings import TransformerEmbedding
from kashgari.embeddings import WordEmbedding
from kashgari.tokenizers import BertTokenizer
from kashgari.tasks.classification import CNN_Attention_Model, BiLSTM_Model
from models.BiLSTM_Att import BiLSTM_Att_Model
from models.RNN_Att import RNN_Att_Model

from sklearn.model_selection import train_test_split
from utils import *


class Trainer:
    def __init__(self, _args):
        self.checkpoint_path = _args.model_folder + '/bert_model.ckpt'
        self.config_path = _args.model_folder + '/bert_config.json'
        self.vocab_path = _args.model_folder + '/vocab.txt'
        self.model_type = _args.model_type
        self.task_type = _args.task_type

    def _embedding(self):
        embedding = TransformerEmbedding(self.vocab_path, self.config_path, self.checkpoint_path,
                                         bert_type=self.model_type)
        return embedding

    def _tokenizing(self, _x):
        tokenizer = BertTokenizer.load_from_vocab_file(self.vocab_path)

        sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in _x]

        return sentences_tokenized

    def train(self, path, **params):
        x_data, y_data = load_data(path, self.task_type)

        embedding = self._embedding()
        x_data = self._tokenizing(x_data)

        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=params["test_size"]
        )
        model = CNN_Attention_Model(embedding)

        if params["validation"]:
            x_train, x_vali, y_train, y_vali = train_test_split(
                x_train, y_train, test_size=params["test_size"]
            )
            model.fit(x_train, y_train, x_vali, y_vali, batch_size=64, epochs=20, callbacks=None, fit_kwargs=None)
        else:
            model.fit(x_train, y_train, batch_size=64, epochs=20, callbacks=None, fit_kwargs=None)
        self.evaluate(model, x_test, y_test)

    @staticmethod
    def evaluate(model, x_test, y_test):
        report = model.evaluate(x_test, y_test, batch_size=64, digits=4, truncating=False)
        print(report)
        # model.save("./models/output")
