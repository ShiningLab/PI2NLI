#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# public
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# private
from src import helper
from src.eval import Evaluator
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
        # results
        self.results_dict = {}
        # model
        self.model = helper.get_model(self.config)
        # datamodule
        self.dm = DataModule(config=self.config)
        # callbacks
        if self.config.key_metric == 'val_f1':
            filename = '{epoch}-{step}-{val_f1:.4f}'
        elif self.config.key_metric == 'val_pos_f1':
            filename = '{epoch}-{step}-{val_pos_f1:.4f}'
        else:
            raise NotImplementedError
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.CKPT_PATH
            , filename=filename
            , monitor=self.config.key_metric
            , mode='max'
            , verbose=True
            , save_last=True
            , save_top_k=1
            )
        early_stop_callback = EarlyStopping(
            monitor=self.config.key_metric
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
            , offline=self.config.offline
            , project=self.config.PROJECT
            , log_model=self.config.log_model
            , entity=self.config.ENTITY
            , save_code=False
            , mode=self.config.wandb_mode
            )
        self.wandb_logger.experiment.config.update(self.config)
        # trainer
        self.trainer = L.Trainer(
            logger = self.wandb_logger
            , callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()]
            , max_epochs=self.config.max_epochs
            , val_check_interval=self.config.val_check_interval
            , enable_checkpointing=True
            , enable_progress_bar=True
            , deterministic=True
            , inference_mode=True
            , profiler=self.config.profiler if self.config.profiler else None
            )

    def train(self):
        self.logger.info('*Configurations:*')
        for k, v in self.config.__dict__.items():
            self.logger.info(f'\t{k}: {v}')
        # 0-shot
        if self.config.test0shot:
            self.logger.info(f"Applying Zero Shot PI2NLI ...")
            predict_dict = self.predict(ckpt_path=None)
            self.results_dict['0shot'] = predict_dict
            self.wandb_logger.log_text(
                key='0shot'
                , columns=list(predict_dict.keys())
                , data=[[predict_dict[k][i] for k in predict_dict] for i in range(self.config.predict_size)]
                )
        # training
        self.logger.info("Start training ...")
        self.trainer.fit(
            model=self.model
            , datamodule=self.dm
            , ckpt_path= 'last' if self.config.load_ckpt else None
            )
        # test
        self.logger.info("Start testing...")
        self.test(ckpt_path='best')
        predict_dict = self.predict(ckpt_path='best')
        # evaluation
        eva = Evaluator(predict_dict['ys_'], predict_dict['ys'])
        eva.get_metrics()
        self.logger.info(eva.info)
        # save results
        self.results_dict['best'] = predict_dict
        self.wandb_logger.log_text(
            key='best'
            , columns=list(predict_dict.keys())
            , data=[[predict_dict[k][i] for k in predict_dict] for i in range(self.config.predict_size)]
            )
        helper.save_pickle(self.results_dict, self.config.RESULTS_PKL)
        self.logger.info('Results saved as {}.'.format(self.config.RESULTS_PKL))
        self.logger.info('Done.')

    def validate(self, ckpt_path=None):
        self.trainer.validate(
            model=self.model
            , datamodule=self.dm
            , ckpt_path=ckpt_path
            , verbose=True
            )

    def test(self, ckpt_path=None):
        self.trainer.test(
            model=self.model
            , datamodule=self.dm
            , ckpt_path=ckpt_path
            , verbose=True
            )

    def predict(self, ckpt_path=None):
        outputs = self.trainer.predict(
            model=self.model
            , datamodule=self.dm
            , ckpt_path=ckpt_path
            , return_predictions=True
            )
        # postprocessing
        outputs_dict = dict()
        for k in outputs[0]:
            outputs_dict[k] = helper.flatten_list([d[k] for d in outputs])
        return outputs_dict