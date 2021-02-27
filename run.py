from train import Trainer
from pretrain import PreTrainer

import argparse


if __name__ == '__main__':
    # model_folder = './models/cased_L-12_H-768_A-12'
    parser = argparse.ArgumentParser(description="pretrain_google for specific data")

    parser.add_argument("--model_folder", type=str, required=True, help="universal pretrained model path")
    parser.add_argument("--model_type", type=str, required=True, default='bert',
                        help="transfer model type, [bert, albert, nezha, gpt2_ml, t5]")
    parser.add_argument("--task_type", type=str, required=True, default='intent',
                        help="specify attributes for predict [intent/influence]")

    args = parser.parse_args()
    assert args.model_type in ['bert', 'albert', 'nezha', 'gpt2_ml', 't5'], + \
        "only support [bert, albert, nezha, gpt2_ml, t5]"
    assert args.task_type in ['intent', 'influence'], "only support [intent, influence]"

    trainer = Trainer(args)
    trainer.train("./datasets/"+args.task_type, test_size=500, validation=False)
