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
from src.utils import helper


class Dataset(Dataset):
    """docstring for Dataset"""
    def __init__(self, mode, tokenizer, config):
        super(Dataset, self).__init__()
        self.mode = mode
        self.config = config
        assert mode in ['train', 'val']
        self.get_data()
        # self.format_data()

    def __len__(self): 
        return self.data_size

    def get_data(self):
        raw_data_path = os.path.join(self.config.DATA_PATH, 'qqp/raw')
        tsv_name = 'dev' if self.mode == 'val' else 'train'
        raw_tsv = os.path.join(raw_data_path, '{}.tsv'.format(tsv_name))
        raw_df = pd.read_csv(raw_tsv, sep='\t')
        # ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
        raw_ds = [raw_df[c].tolist() for c in raw_df.columns]
        self.qs1_list, self.qs2_list, self.ys_list = raw_ds[3:]
        self.data_size = len(self.qs1_list)

    def get_train_instances(self, q1, q2, y):
        # id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        if y:  # positive instance
            return (q1, q2, 0), (q2, q1, 0)
        else:  # negative instance
            return (q1, q2, random.randint(1, 2)), (q2, q1, random.randint(1, 2))

    def collate_fn(self, tokenizer, training, config, data):
        # a customized collate function used in the data loader
        if training:
            data = helper.flatten_list(data)
            raw_qs1, raw_qs2, raw_ys = zip(*data)
            xs_inputs = tokenizer.batch_encode_plus(
                list(zip(raw_qs1, raw_qs2))
                , add_special_tokens=True
                , return_tensors='pt'
                , padding='max_length'
                , truncation=True
                , max_length=config.max_length
             )
            return xs_inputs, torch.LongTensor(raw_ys)
        else:
            raw_qs1, raw_qs2, raw_ys = zip(*data)
            xs_inputs_0 = tokenizer.batch_encode_plus(
                list(zip(raw_qs1, raw_qs2))
                , add_special_tokens=True
                , return_tensors='pt'
                , padding='max_length'
                , truncation=True
                , max_length=config.max_length
             )
            xs_inputs_1 = tokenizer.batch_encode_plus(
                list(zip(raw_qs2, raw_qs1))
                , add_special_tokens=True
                , return_tensors='pt'
                , padding='max_length'
                , truncation=True
                , max_length=config.max_length
             )
            return xs_inputs_0, xs_inputs_1, torch.LongTensor(raw_ys)

    def __getitem__(self, idx):
        q1, q2, y = self.qs1_list[idx], self.qs2_list[idx], self.ys_list[idx]
        if self.mode == 'train':
            return self.get_train_instances(q1, q2, y)
        return q1, q2, y