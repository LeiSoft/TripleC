from train import Trainer
from pretrain import PreTrainer

import argparse


if __name__ == '__main__':
    # model_folder = './models/cased_L-12_H-768_A-12'
    parser = argparse.ArgumentParser(description="training model")

    parser.add_argument("--model_folder", type=str, help="universal pretrained model path")
    parser.add_argument("--model_type", type=str, default='bert',
                        help="transfer model type, [bert, w2v, albert, nezha, gpt2_ml, t5]")
    parser.add_argument("--task_type", type=str, required=True, default='intent',
                        help="specify attributes for predict [intent/influence]")

    args = parser.parse_args()
    assert args.model_type in ['bert', 'w2v', 'albert', 'nezha', 'gpt2_ml', 't5'], + \
        "only support [bert, w2v, albert, nezha, gpt2_ml, t5]"
    assert args.task_type in ['intent', 'influence'], "only support [intent, influence]"

    trainer = Trainer(args)
    trainer.train("./datasets/"+args.task_type+"/train.tsv", test_size=0.2, validation=False)
