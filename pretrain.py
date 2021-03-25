from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
import tensorflow as tf
import tensorflow_datasets as tfds
from kashgari.embeddings import BertEmbedding
from kashgari.processors import SequenceProcessor
import argparse
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling, AlbertForMaskedLM, AlbertConfig

from train import Trainer
from utils import *


class PreTrainer(Trainer):
    def __init__(self, _args):
        super().__init__(_args)
        self.datasets_path = "./datasets/"+_args.task_type+"/train.tsv"
        self.model_path = _args.model_path
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased", max_len=512)

    def _generate_ds(self):
        x_data, y_data = load_data(self.datasets_path)
        for i in range(len(y_data)):
            y_data[i] = int(y_data[i])

        x_data = [" ".join(seq) for seq in x_data]

        dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path="./datasets/corpora.txt",
            block_size=256,
        )

        # embeder = self._embedding()
        # processor = SequenceProcessor()
        # embeder.setup_text_processor(processor)
        # x_data = embeder.embed(x_data)
        #
        # def _tensor():
        #     for a, b in zip(x_data, y_data):
        #         yield (a, [b])
        #
        # # shape = (tf.TensorShape([len(x_data[0]), len(x_data[0][0])]), tf.TensorShape([len(x_data[0]), 1]))
        # dataset = tf.data.Dataset.from_generator(generator=_tensor, output_types=(tf.int32, tf.int32))

        return dataset

    def train(self, path, **params):
        # model = BertForSequenceClassification.from_pretrained("bert-large-cased")
        config = AlbertConfig(
            vocab_size=1359,
            embedding_size=256,
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
        )
        model = AlbertForMaskedLM(config=config)

        training_args = TrainingArguments(
            output_dir='./models/bert4tc/'+args.task_type,  # output directory
            num_train_epochs=3,  # total # of training epochs
            per_device_train_batch_size=1,  # batch size per device during training
            # per_device_eval_batch_size=16,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            # logging_dir='./logs',  # directory for storing logs
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            data_collator=data_collator,
            train_dataset=self._generate_ds(),  # tensorflow_datasets training dataset
            # eval_dataset=tfds_test_dataset  # tensorflow_datasets evaluation dataset
        )

        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pretrain my model")

    parser.add_argument("--task_type", type=str, default='intent',
                        help="specify attributes for predict [intent/influence]")
    parser.add_argument("--model_path", type=str, help="specify pretrained model path",
                        default="./models/cased_L-24_H-1024_A-16")
    parser.add_argument("--model_folder", type=str, help="universal pretrained model path")
    parser.add_argument("--model_type", type=str, default='bert',
                        help="transfer model type, [bert, w2v]")

    args = parser.parse_args()
    assert args.task_type in ['intent', 'influence'], "only support [intent, influence]"
    assert args.model_type in ['bert', 'w2v'], + \
        "only support [bert, w2v]"

    pretrainer = PreTrainer(args)
    pretrainer.train("./datasets/"+args.task_type+"/train.tsv")