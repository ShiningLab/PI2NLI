#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os
# public
import torch
from lightning.pytorch import seed_everything
# private
from config import Config
from src import helper
from src.trainer import LitTrainer


class NLIer(object):
    """docstring for NLIer"""
    def __init__(self):
        super(NLIer, self).__init__()
        self.config = Config()
        self.update_config()
        self.initialize()

    def update_config(self):
        # setup device
        self.config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def initialize(self):
        # results
        self.results_dict = {}
        # get trainer
        self.trainer = LitTrainer(self.config)
        # setup random seed
        seed_everything(self.config.seed, workers=True)
        # enable tokenizer multi-processing
        if self.config.num_workers > 0:
            os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        # others
        torch.set_float32_matmul_precision('high')

    def train(self):
        # self.results_dict['0shot'] = self.trainer.validate()
        # ckpt_path = os.path.join(self.config.CKPT_PATH, 'epoch=7-step=90960.ckpt')
        # self.results_dict['best'] = self.trainer.validate(ckpt_path)
        # save results
        # helper.save_pickle(self.config.RESOURCE_PKL, self.results_dict)
        self.trainer.train()

def main() -> None:
    nlier = NLIer()
    nlier.train()

if __name__ == '__main__':
      main()