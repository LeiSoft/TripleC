from transformers import TFTrainer, TFTrainingArguments, BertTokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import argparse
from transformers import TFBertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

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

        x_data = ["[CLS] "+" ".join(seq)+" [SEP]" for seq in x_data]

        tokenizer = BertTokenizer.from_pretrained("D:/python/myLibs/bert-base-cased")
        tokenized_texts = [tokenizer.tokenize(sent) for sent in x_data]

        # Set the maximum sequence length.
        MAX_LEN = 128
        # Pad our input tokens
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        # Create attention masks
        attention_masks = []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                            random_state=2018,
                                                                                            test_size=0.1)
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                               random_state=2018, test_size=0.1)

        # Convert all of our data into torch tensors, the required datatype for our model
        train_inputs = tf.constant(train_inputs)
        validation_inputs = tf.constant.tensor(validation_inputs)
        train_labels = tf.constant.tensor(train_labels)
        validation_labels = tf.constant.tensor(validation_labels)
        train_masks = tf.constant.tensor(train_masks)
        validation_masks = tf.constant.tensor(validation_masks)

        # Select a batch size for training.
        batch_size = 32

        # Create an iterator of our data with torch DataLoader
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        return dataset

    def train(self, path, **params):
        model = TFBertModel.from_pretrained("D:/python/myLibs/bert-base-cased")

        training_args = TFTrainingArguments(
            output_dir='./models/bert4tc/'+args.task_type,  # output directory
            num_train_epochs=3,  # total # of training epochs
            per_device_train_batch_size=1,  # batch size per device during training
            # per_device_eval_batch_size=16,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            # logging_dir='./logs',  # directory for storing logs
        )

        trainer = TFTrainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=self._generate_ds(),  # tensorflow_datasets training dataset
            # eval_dataset=tfds_test_dataset  # tensorflow_datasets evaluation dataset
        )

        trainer.train()
        trainer.save_model()


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