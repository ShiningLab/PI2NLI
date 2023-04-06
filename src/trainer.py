#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
from transformers import AutoTokenizer
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# private
from src import helper
from src.datamodule import DataModule


class LitTrainer(object):
    """docstring for LitTrainer"""
    def __init__(self, config, **kwargs):
        super(LitTrainer, self).__init__()
        self.config = config
        self.update_config(**kwargs)
        self.initialize()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def initialize(self):
        # model
        self.model = helper.get_model(self.config)
        # datamodule
        self.dm = DataModule(
            tokenizer = AutoTokenizer.from_pretrained(self.config.LM_PATH)
            , config = self.config
            )
        # callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.CKPT_PATH
            , filename='{epoch}-{step}-{val_acc:.2f}'
            , monitor='val_acc'
            , mode='max'
            , verbose=True
            , save_last=True
            , save_top_k=1
            )
        early_stop_callback = EarlyStopping(
            monitor='val_acc'
            , min_delta=.0
            , patience=self.config.patience
            , verbose=True
            , mode='max'
            )
        # logger
        self.logger = helper.init_logger(self.config)
        self.logger.info('Logger initialized.')
        self.wandb_logger = WandbLogger(
            name=self.config.NAME
            , save_dir=self.config.LOG_PATH
            , offline=self.config.OFFLINE
            , project=self.config.PROJECT
            , log_model=False if self.config.OFFLINE else 'all'
            , entity=self.config.ENTITY
            )
        # trainer
        self.trainer = pl.Trainer(
            logger = self.wandb_logger
            , callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()]
            , max_epochs=self.config.max_epochs
            , enable_checkpointing=True
            , enable_progress_bar=True
            , deterministic=True
            , inference_mode=True
            )

    def train(self):
        self.trainer.fit(
            model=self.model
            , datamodule=self.dm
            , ckpt_path= 'last' if self.config.load_ckpt else None
            )

    def validate(self, ckpt_path=None):
        # validation
        outputs = self.trainer.predict(
            model=self.model
            , datamodule=self.dm
            , ckpt_path='best'
            , verbose=True
            )
        # postprocessing
        outputs_dict = dict()
        for k in outputs[0]:
            outputs_dict[k] = helper.flatten_list([d[k] for d in outputs] )
        return outputs_dict