import argparse
import os
from transformers import BertTokenizerFast, BertForMaskedLM
from transformers import LineByLineTextDataset, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from convert_pytorch_checkpoint_to_tf2 import *


class PreTrainer:
    def __init__(self, _args):
        self._args = _args
        self.model_path = _args.model_path

    def train(self):
        tokenizer = BertTokenizerFast.from_pretrained(self.model_path, max_len=256)

        model = BertForMaskedLM.from_pretrained(self.model_path)
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path="datasets/corpora.txt",
            block_size=128,
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
        training_args = TrainingArguments(
            output_dir="./models/bert4tc",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=32,
            save_steps=500,
            save_total_limit=5,
            prediction_loss_only=True,
            do_train=True,
            do_eval=True,
            load_best_model_at_end=True,
            eval_steps=500,
            learning_rate=1e-4,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        trainer.train()
        trainer.save_model("./models/bert4tc")

    @staticmethod
    def convert_torch2tf(pt_model: str):
        model = BertForMaskedLM.from_pretrained(pt_model)
        convert_pytorch_checkpoint_to_tf2(model=model, ckpt_dir="./models/bert4tc_tf", model_name="bert_model")
        os.system("cp "+pt_model+"/config.json ./models/bert4tc_tf/bert_config.json")

    @staticmethod
    def get_add_vocab(path):
        vocab = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                for word in line.strip(" "):
                    vocab[word] = vocab.get(word, 0) + 1
        res = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:1]
        return [it[0] for it in res]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pretrain my model")

    parser.add_argument("--model_path", type=str, help="specify pretrained model path",
                        default="./models/bert-large-cased")
    parser.add_argument("--model_type", type=str, default='bert',
                        help="specify model type, [bert]")

    args = parser.parse_args()

    pretrainer = PreTrainer(args)
    pretrainer.train()

    pretrainer.convert_torch2tf("./models/bert4tc")
