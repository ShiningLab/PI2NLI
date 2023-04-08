#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import torch
from transformers import AutoModelForSequenceClassification
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score


class PI2NLIClassifier(pl.LightningModule):
    """docstring for PI2NLIClassifier"""
    def __init__(self, config, **kwargs):
        super(PI2NLIClassifier, self).__init__()
        self.config = config
        self.update_config(**kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.LM_PATH
            )
        self.val_acc, self.val_f1 = BinaryAccuracy(), BinaryF1Score()
        self.test_acc, self.test_f1 = BinaryAccuracy(), BinaryF1Score()
        self.save_hyperparameters()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        loss = self.model(**xs, labels=ys).loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        _, batch = batch
        xs0, xs1, ys = batch
        outputs0 = self.model(**xs0, labels=None)
        outputs1 = self.model(**xs1, labels=None)
        ys0_ = outputs0.logits.softmax(dim=1).argmax(dim=1)
        ys1_ = outputs1.logits.softmax(dim=1).argmax(dim=1)
        ys_ = ((ys0_ == ys1_) * (ys0_ == self.config.ENTAILMENT)).int()
        for metric, metric_f in zip(['val_acc', 'val_f1'], [self.val_acc, self.val_f1]):
            metric_v = metric_f(ys_, ys).item()
            self.log(metric, metric_v, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        _, batch = batch
        xs0, xs1, ys = batch
        outputs0 = self.model(**xs0, labels=None)
        outputs1 = self.model(**xs1, labels=None)
        ys0_ = outputs0.logits.softmax(dim=1).argmax(dim=1)
        ys1_ = outputs1.logits.softmax(dim=1).argmax(dim=1)
        ys_ = ((ys0_ == ys1_) * (ys0_ == self.config.ENTAILMENT)).int()
        for metric, metric_f in zip(['test_acc', 'test_f1'], [self.test_acc, self.test_f1]):
            metric_v = metric_f(ys_, ys).item()
            self.log(metric, metric_v, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        raw_batch, batch = batch
        raw_xs0, raw_xs1, ys = raw_batch
        xs0, xs1, _ = batch
        outputs0 = self.model(**xs0, labels=None)
        outputs1 = self.model(**xs1, labels=None)
        ys0_ = outputs0.logits.softmax(dim=1).argmax(dim=1)
        ys1_ = outputs1.logits.softmax(dim=1).argmax(dim=1)
        ys_ = ((ys0_ == ys1_) * (ys0_ == self.config.ENTAILMENT)).int()
        return {'raw_xs0': raw_xs0, 'raw_xs1': raw_xs1, 'ys': ys, 'ys0_': ys0_, 'ys1_': ys1_, 'ys_': ys_}

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)]
                , "weight_decay": self.config.weight_decay
                }
            , {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)]
                , "weight_decay": 0.0
                }
            ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters
            , lr=self.config.learning_rate
            , eps=self.config.adam_epsilon
            )
        return optimizer


class PIClassifier(pl.LightningModule):
    """docstring for PIClassifier"""
    def __init__(self, config, **kwargs):
        super(PIClassifier, self).__init__()
        self.config = config
        self.update_config(**kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.LM_PATH
            , num_labels=2
            , id2label={0: 'negative', 1: 'positive'}
            , label2id={'negative': 0, 'positive': 1}
            )
        self.val_acc, self.val_f1 = BinaryAccuracy(), BinaryF1Score()
        self.test_acc, self.test_f1 = BinaryAccuracy(), BinaryF1Score()
        self.save_hyperparameters()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        loss = self.model(**xs, labels=ys).loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        _, batch = batch
        xs, ys = batch
        outputs = self.model(**xs, labels=None)
        ys_ = outputs.logits.softmax(dim=1).argmax(dim=1)
        for metric, metric_f in zip(['val_acc', 'val_f1'], [self.val_acc, self.val_f1]):
            metric_v = metric_f(ys_, ys).item()
            self.log(metric, metric_v)

    def test_step(self, batch, batch_idx):
        _, batch = batch
        xs, ys = batch
        outputs = self.model(**xs, labels=None)
        ys_ = outputs.logits.softmax(dim=1).argmax(dim=1)
        for metric, metric_f in zip(['test_acc', 'test_f1'], [self.test_acc, self.test_f1]):
            metric_v = metric_f(ys_, ys).item()
            self.log(metric, metric_v, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        raw_batch, batch = batch
        xs0, xs1, ys = raw_batch
        xs, _ = batch
        outputs = self.model(**xs, labels=None)
        ys_ = outputs.logits.softmax(dim=1).argmax(dim=1)
        return {'xs0': xs0, 'xs1': xs1, 'ys': ys, 'ys_': ys_}

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)]
                , "weight_decay": self.config.weight_decay
                }
            , {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)]
                , "weight_decay": 0.0
                }
            ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters
            , lr=self.config.learning_rate
            , eps=self.config.adam_epsilon
            )
        return optimizer