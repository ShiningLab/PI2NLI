#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

# dependency
# public
import torch
import wandb
from torchmetrics.functional.classification import binary_accuracy, binary_f1_score


class Evaluator(object):
    """docstring for Evaluator"""
    def __init__(self, ys_, ys):
        super(Evaluator, self).__init__()
        self.preds = torch.LongTensor(ys_)
        self.target = torch.LongTensor(ys)

    def get_metrics(self):
        # accuracy
        self.acc = binary_accuracy(self.preds, self.target)
        self.pos_acc = binary_accuracy(self.preds, self.target, ignore_index=0)
        self.neg_acc = binary_accuracy(self.preds, self.target, ignore_index=1)
        # f1 score
        self.f1 = binary_f1_score(self.preds, self.target)
        self.pos_f1 = binary_f1_score(self.preds, self.target, ignore_index=0)
        # update eval info
        self.get_info()
        # update logger
        try:
            wandb.log(self.metrics_dict)
        except:
            pass

    def get_info(self):
        # format
        self.metrics_dict =  {
            'pred_acc': self.acc
            , 'pred_pos_acc': self.pos_acc
            , 'pred_neg_acc': self.neg_acc
            , 'pred_f1': self.f1
            , 'pred_pos_f1': self.pos_f1
            }
        # get info
        self.info = '|'
        for k, v in self.metrics_dict.items():
            self.info += '{}:{:.4f}|'.format(k, v*100)