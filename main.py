#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import torch
# private
from config import Config
from src.utils import helper
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
        # setup random seed
        helper.set_seed(self.config.seed)

    def train(self):
        trainer = LitTrainer(self.config)
        trainer.train()

def main() -> None:
    nlier = NLIer()
    nlier.train()

if __name__ == '__main__':
      main()