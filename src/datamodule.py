#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
from functools import partial
# public
import lightning.pytorch as pl
from torch.utils.data import DataLoader
# private
from src import helper


class DataModule(pl.LightningDataModule):
    """docstring for DataModule"""
    def __init__(self, tokenizer, config):
        super(DataModule, self).__init__()
        self.dataset = helper.get_dataset(config)
        self.tokenizer = tokenizer
        self.config = config

    def setup(self, stage: str):
        match stage:
            case 'fit':
                self.train_dataset = self.dataset('train', self.tokenizer, self.config, samplesize=100)
                self.val_dataset = self.dataset('val', self.tokenizer, self.config, samplesize=100)
            case 'predict':
                self.predict_dataset  = self.dataset('val', self.tokenizer, self.config, samplesize=100)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset
            , batch_size=self.config.train_batch_size
            , collate_fn=partial(self.train_dataset.collate_fn, self.tokenizer, True, self.config)
            , shuffle=True
            , num_workers=self.config.num_workers
            , pin_memory=True
            , drop_last=True
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset
            , batch_size=self.config.train_eval_size
            , collate_fn=partial(self.val_dataset.collate_fn, self.tokenizer, False, self.config)
            , shuffle=False
            , num_workers=self.config.num_workers
            , pin_memory=True
            , drop_last=False
            )

    def predict_dataloader(self):
        return torch_data.DataLoader(
            self.predict_dataset
            , batch_size=self.config.train_eval_size
            , collate_fn=partial(self.predict_dataset.collate_fn, self.tokenizer, False, self.config)
            , shuffle=False
            , num_workers=self.config.num_workers
            , pin_memory=True
            , drop_last=False
            )