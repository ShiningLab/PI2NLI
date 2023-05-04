#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# built-in
import random
# public
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
# private
from src import helper


class PI2NLIDataset(Dataset):
    """docstring for PI2NLIDataset"""
    def __init__(self, mode, config, samplesize=None):
        super(PI2NLIDataset, self).__init__()
        assert mode in ['train', 'val', 'test']
        assert config.method in ['mut_pi2nli', 'asym_pi2nli']
        assert config.nli_mode in ['rand', 'nli']
        self.mode = mode
        self.config = config
        self.samplesize = samplesize
        self.tokenizer=AutoTokenizer.from_pretrained(config.LM_PATH)
        self.get_data()

    def __len__(self): 
        return self.data_size

    def get_data(self):
        data_dict = helper.load_pickle(self.config.DATA_PKL)
        if self.mode == 'train':
            data_dict = data_dict[self.config.method]  # mut_pi2nli or asym_pi2nli
        else:
            data_dict = data_dict[self.mode]  # val or test
        self.xs0_list = data_dict['xs0']
        self.xs1_list = data_dict['xs1']
        if self.mode == 'train':
             self.ys_list = data_dict[self.config.nli_mode]  # rand or nli
        else:
            self.ys_list = data_dict['ys']
        self.data_size = len(self.ys_list)
        if self.samplesize:
            idxes = list(range(self.data_size))
            random.shuffle(idxes)
            idxes = idxes[:self.samplesize]
            self.xs0_list = [self.xs0_list[i] for i in idxes]
            self.xs1_list = [self.xs1_list[i] for i in idxes]
            self.ys_list = [self.ys_list[i] for i in idxes]
            self.data_size = self.samplesize

    def collate_fn(self, training, config, data):
        # a customized collate function used in the data loader
        if training:
            raw_xs0, raw_xs1, raw_ys = zip(*data)
            xs_inputs = self.tokenizer.batch_encode_plus(
                list(zip(raw_xs0, raw_xs1))
                , add_special_tokens=True
                , return_tensors='pt'
                , padding='max_length'
                , truncation=True
                , max_length=config.max_length
             )
            return xs_inputs, torch.LongTensor(raw_ys)
        else:
            # x0 -> x1
            raw_xs0, raw_xs1, raw_ys = zip(*data)
            xs_inputs_0 = self.tokenizer.batch_encode_plus(
                list(zip(raw_xs0, raw_xs1))
                , add_special_tokens=True
                , return_tensors='pt'
                , padding='max_length'
                , truncation=True
                , max_length=config.max_length
             )
            # x1 -> x0
            xs_inputs_1 = self.tokenizer.batch_encode_plus(
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
        return self.xs0_list[idx], self.xs1_list[idx], self.ys_list[idx]


class PIDataset(Dataset):
    """docstring for PIDataset"""
    def __init__(self, mode, config, samplesize=None):
        super(PIDataset, self).__init__()
        assert mode in ['train', 'val', 'test']
        assert config.method in ['pi']
        assert config.nli_mode in ['none']
        self.mode = mode
        self.config = config
        self.samplesize = samplesize
        self.tokenizer=AutoTokenizer.from_pretrained(config.LM_PATH)
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
            self.data_size = self.samplesize

    def collate_fn(self, training, config, data):
        # a customized collate function used in the data loader
        raw_xs0, raw_xs1, raw_ys = zip(*data)
        xs_inputs = self.tokenizer.batch_encode_plus(
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