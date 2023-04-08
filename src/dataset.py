#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# built-in
import os, random
# public
import torch
import pandas as pd
from torch.utils.data import Dataset
# private
from src import helper


class PI2NLIDataset(Dataset):
    """docstring for PI2NLIDataset"""
    def __init__(self, mode, config, samplesize=None):
        super(PI2NLIDataset, self).__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.config = config
        self.samplesize = samplesize
        self.get_data()

    def __len__(self): 
        return self.data_size

    def get_data(self):
        data_dict = helper.load_pickle(self.config.DATA_PKL)
        self.xs0_list = data_dict[self.mode]['xs0']
        self.xs1_list = data_dict[self.mode]['xs1']
        self.ys_list = data_dict[self.mode]['ys']
        self.data_size = len(self.ys_list)
        if self.samplesize:
            idxes = list(range(self.data_size))
            random.shuffle(idxes)
            idxes = idxes[:self.samplesize]
            self.xs0_list = [self.xs0_list[i] for i in idxes]
            self.xs1_list = [self.xs1_list[i] for i in idxes]
            self.ys_list = [self.ys_list[i] for i in idxes]
            self.data_size = len(self.ys_list)

    def get_train_instances(self, x0, x1, y):
        # id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        if y:  # positive instance
            return (x0, x1, 0), (x0, x1, 0)
        else:  # negative instance
            return (x0, x1, random.randint(1, 2)), (x1, x0, random.randint(1, 2))

    def collate_fn(self, tokenizer, training, config, data):
        # a customized collate function used in the data loader
        if training:
            data = helper.flatten_list(data)
            raw_xs0, raw_xs1, raw_ys = zip(*data)
            xs_inputs = tokenizer.batch_encode_plus(
                list(zip(raw_xs0, raw_xs1))
                , add_special_tokens=True
                , return_tensors='pt'
                , padding='max_length'
                , truncation=True
                , max_length=config.max_length
             )
            return xs_inputs, torch.LongTensor(raw_ys)
        else:
            raw_xs0, raw_xs1, raw_ys = zip(*data)
            xs_inputs_0 = tokenizer.batch_encode_plus(
                list(zip(raw_xs0, raw_xs1))
                , add_special_tokens=True
                , return_tensors='pt'
                , padding='max_length'
                , truncation=True
                , max_length=config.max_length
             )
            xs_inputs_1 = tokenizer.batch_encode_plus(
                list(zip(raw_xs1, raw_xs0))
                , add_special_tokens=True
                , return_tensors='pt'
                , padding='max_length'
                , truncation=True
                , max_length=config.max_length
             )
            return (raw_xs0, raw_xs1, raw_ys), \
            (xs_inputs_0, xs_inputs_1, torch.LongTensor(raw_ys))

    def __getitem__(self, idx):
        x0, x1, y = self.xs0_list[idx], self.xs1_list[idx], self.ys_list[idx]
        if self.mode == 'train':
            # PI2NLI
            return self.get_train_instances(x0, x1, y)
        # PI
        return x0, x1, y


class PIDataset(Dataset):
    """docstring for PIDataset"""
    def __init__(self, mode, config, samplesize=None):
        super(PIDataset, self).__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.config = config
        self.samplesize = samplesize
        self.get_data()

    def __len__(self): 
        return self.data_size

    def get_data(self):
        data_dict = helper.load_pickle(self.config.DATA_PKL)
        self.xs0_list = data_dict[self.mode]['xs0']
        self.xs1_list = data_dict[self.mode]['xs1']
        self.ys_list = data_dict[self.mode]['ys']
        self.data_size = len(self.ys_list)
        if self.samplesize:
            idxes = list(range(self.data_size))
            random.shuffle(idxes)
            idxes = idxes[:self.samplesize]
            self.xs0_list = [self.xs0_list[i] for i in idxes]
            self.xs1_list = [self.xs1_list[i] for i in idxes]
            self.ys_list = [self.ys_list[i] for i in idxes]
            self.data_size = len(self.ys_list)

    def collate_fn(self, tokenizer, training, config, data):
        # a customized collate function used in the data loader
        raw_xs0, raw_xs1, raw_ys = zip(*data)
        xs_inputs = tokenizer.batch_encode_plus(
            list(zip(raw_xs0, raw_xs1))
            , add_special_tokens=True
            , return_tensors='pt'
            , padding='max_length'
            , truncation=True
            , max_length=config.max_length
         )
        if training:
            return xs_inputs, torch.LongTensor(raw_ys)
        else:
            return (raw_xs0, raw_xs1, raw_ys), \
            (xs_inputs, torch.LongTensor(raw_ys))

    def __getitem__(self, idx):
        return self.xs0_list[idx], self.xs1_list[idx], self.ys_list[idx]