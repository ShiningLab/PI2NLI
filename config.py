#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os, argparse
# private
from src import helper


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--load_ckpt', type=helper.str2bool, default=False)
    # pi for training PI models then testing on PI
    # pi2nli for training NLI models then testing on PI
    parser.add_argument('--task', type=str, default='pi2nli')
    # qqp for Quora Question Pairs
    parser.add_argument('--data', type=str, default='qqp')
    # model
    parser.add_argument('--max_length', type=int, default=156)
    # NLI
    # roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
    # PI
    # roberta-large
    parser.add_argument('--model', type=str, default='roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')
    # training
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=-1)  # to enable infinite training
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    # evaluation
    parser.add_argument('--patience', type=int, default=6)
    # logger
    parser.add_argument('--offline', type=helper.str2bool, default=False)
    parser.add_argument('--log_model', type=helper.str2bool, default=False)
    # save as argparse space
    return parser.parse_known_args()[0]

class Config(object):
    """docstring for Config"""
    def __init__(self):
        super(Config, self).__init__()
        self.update_config(**vars(init_args()))

    def update_config(self, **kwargs):
        # load config from parser
        for k,v in kwargs.items():
            setattr(self, k, v)
        # I/O
        # self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.CURR_PATH = './'
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        os.makedirs(self.DATA_PATH, exist_ok=True)
        self.DATA_PKL = os.path.join(self.DATA_PATH, f'{self.data}.pkl')
        # language model
        self.LM_PATH = os.path.join(self.RESOURCE_PATH, 'lm', self.model)
        # checkpoints
        self.CKPT_PATH = os.path.join(
            self.RESOURCE_PATH, 'ckpts', self.task, self.data, self.model, str(self.seed)
            )
        os.makedirs(self.CKPT_PATH, exist_ok=True)
        self.CKPT_LAST = os.path.join(self.CKPT_PATH, 'last.ckpt')
        # log
        self.ENTITY = 'mrshininnnnn'
        self.PROJECT = 'PI2NLI'
        self.NAME = f'{self.task}-{self.data}-{self.model}-{self.seed}'
        self.LOG_PATH = os.path.join(
            self.RESOURCE_PATH, 'log', self.task, self.data, self.model, str(self.seed)
            )
        os.makedirs(self.LOG_PATH, exist_ok=True)
        self.LOG_TXT = os.path.join(self.LOG_PATH, 'console_log.txt')
        os.remove(self.LOG_TXT) if os.path.exists(self.LOG_TXT) else None
        # results
        self.RESULTS_PATH = os.path.join(
            self.RESOURCE_PATH, 'results', self.task, self.data, self.model,
            )
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        self.RESULTS_PKL = os.path.join(self.RESULTS_PATH, f'{self.seed}.pkl')
        # NLI
        # id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        self.ENTAILMENT = 0