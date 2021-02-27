
from train import Trainer


class PreTrainer(Trainer):
    def __init__(self, _args):
        super().__init__(_args)

    def train(self, path, **params):
        pass


