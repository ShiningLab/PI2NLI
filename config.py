#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# built-in
import os, argparse
# private
from src import helper


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # pi for paraphrase identification
    # mut_pi2nli for mutual entailment
    # asym_pi2nli for asymmetric entailment
    parser.add_argument('--method', type=str, default='pi')
    # pit for Paraphrase and Semantic Similarity in Twitter
    # qqp for Quora Question Pairs
    # mrpc for Microsoft Research Paraphrase Corpus
    # paws_qqp for Paraphrase Adversaries from Word Scrambling QQP
    # paws_wiki for Paraphrase Adversaries from Word Scrambling WIKI
    # parade for Paraphrase Identification Requiring Computer Science Domain Knowledge
    # all
    parser.add_argument('--data', type=str, default='pit')
    # model
    # roberta-large
    # xlnet-large-cased
    # roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
    # xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli
    parser.add_argument('--model', type=str, default='roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')
    # if the classification head is initialized from scratch
    parser.add_argument('--init_classifier', type=helper.str2bool, default=True)
    parser.add_argument('--test0shot', type=helper.str2bool, default=False)
    parser.add_argument('--max_length', type=int, default=156)
    parser.add_argument('--load_ckpt', type=helper.str2bool, default=False)
    # training
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=1)  # -1 to enable infinite training
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    # evaluation
    parser.add_argument('--key_metric', type=str, default='val_f1')
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--val_check_interval', type=float, default=1.0)  # 0.5 for QQP
    # trainer
    # (str, optional) Can be 'simple' or 'advanced'. Defaults to ''.
    parser.add_argument('--profiler', type=str, default='') 
    # logger
    parser.add_argument('--offline', type=helper.str2bool, default=True) # True for development
    # (str, optional) Can be 'online', 'offline' or 'disabled'. Defaults to online.
    parser.add_argument('--wandb_mode', type=str, default='disabled')  # disabled for testing code
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
        # update config
        if self.data == 'qqp':
            self.val_check_interval = 0.5
        else:
            self.val_check_interval = 1.0
        # I/O
        self.CURR_PATH = './'
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        os.makedirs(self.DATA_PATH, exist_ok=True)
        self.DATA_PKL = os.path.join(self.DATA_PATH, f'{self.data}.pkl')
        # language model
        self.LM_PATH = os.path.join(self.RESOURCE_PATH, 'lm', self.model)
        # checkpoints
        self.CKPT_PATH = os.path.join(
            self.RESOURCE_PATH, 'ckpts', self.method, self.data, self.model, str(self.seed)
            )
        os.makedirs(self.CKPT_PATH, exist_ok=True)
        self.CKPT_LAST = os.path.join(self.CKPT_PATH, 'last.ckpt')
        # log
        self.ENTITY = 'ENTITY'
        self.PROJECT = 'PI2NLI'
        self.NAME = f'{self.method}-{self.data}-{self.model}-{self.seed}'
        self.LOG_PATH = os.path.join(
            self.RESOURCE_PATH, 'log', self.method, self.data, self.model, str(self.seed)
            )
        os.makedirs(self.LOG_PATH, exist_ok=True)
        self.LOG_TXT = os.path.join(self.LOG_PATH, 'console_log.txt')
        os.remove(self.LOG_TXT) if os.path.exists(self.LOG_TXT) else None
        # results
        self.RESULTS_PATH = os.path.join(
            self.RESOURCE_PATH, 'results', self.method, self.data, self.model,
            )
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        self.RESULTS_PKL = os.path.join(self.RESULTS_PATH, f'{self.seed}.pkl')
        # NLI
        # id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        self.ENTAILMENT = 0