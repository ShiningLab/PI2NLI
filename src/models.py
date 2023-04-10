#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
import lightning.pytorch as pl
from torchmetrics.classification import Accuracy, BinaryAccuracy, F1Score, BinaryF1Score


class PI2NLIClassifier(pl.LightningModule):
    """docstring for PI2NLIClassifier"""
    def __init__(self, config, **kwargs):
        super(PI2NLIClassifier, self).__init__()
        self.config = config
        self.update_config(**kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.LM_PATH
            )
        self.train_loss_list = []
        self.train_acc = Accuracy(task='multiclass', num_classes=3)
        self.train_f1 = F1Score(task='multiclass', num_classes=3)
        self.val_acc, self.val_f1 = BinaryAccuracy(), BinaryF1Score()
        self.test_acc, self.test_f1 = BinaryAccuracy(), BinaryF1Score()
        self.save_hyperparameters()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        outputs = self.model(**xs, labels=ys)
        loss = outputs.loss.item()
        self.train_loss_list.append(loss)
        ys_ = outputs.logits.softmax(dim=1).argmax(dim=1)
        self.log('train_step_loss', loss, prog_bar=True)
        self.train_acc.update(ys_, ys)
        self.train_f1.update(ys_, ys)
        return outputs.loss

    def on_train_epoch_end(self):
        self.log_dict({
            'train_epoch_loss': np.mean(self.train_loss_list, dtype='float32')
            , 'train_acc': self.train_acc.compute()
            , 'train_f1': self.train_f1.compute()
            })
        self.train_loss_list = []
        self.train_acc.reset()
        self.train_f1.reset()

    def pi2nli(self, xs0, xs1):
        ys0_ = self.model(**xs0, labels=None).logits.softmax(dim=1).argmax(dim=1)
        ys1_ = self.model(**xs1, labels=None).logits.softmax(dim=1).argmax(dim=1)
        ys_ = ((ys0_ == ys1_) * (ys0_ == self.config.ENTAILMENT)).int()
        return ys0_, ys1_, ys_

    def validation_step(self, batch, batch_idx):
        _, batch = batch
        xs0, xs1, ys = batch
        _, _, ys_ = self.pi2nli(xs0, xs1)
        self.val_acc.update(ys_, ys)
        self.val_f1.update(ys_, ys)

    def on_validation_epoch_end(self):
        self.log_dict({
            'val_acc': self.val_acc.compute()
            , 'val_f1': self.val_f1.compute()
            })
        self.val_acc.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        _, batch = batch
        xs0, xs1, ys = batch
        _, _, ys_ = self.pi2nli(xs0, xs1)
        self.test_acc.update(ys_, ys)
        self.test_f1.update(ys_, ys)

    def on_test_epoch_end(self):
        self.log_dict({
            'test_acc': self.test_acc.compute()
            , 'test_f1': self.test_f1.compute()
            })
        self.test_acc.reset()
        self.test_f1.reset()

    def predict_step(self, batch, batch_idx):
        (raw_xs0, raw_xs1, ys), (xs0, xs1, _) = batch
        ys0_, ys1_, ys_ = self.pi2nli(xs0, xs1)
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
        self.train_loss_list = []
        self.train_acc, self.train_f1 = BinaryAccuracy(), BinaryF1Score()
        self.val_acc, self.val_f1 = BinaryAccuracy(), BinaryF1Score()
        self.test_acc, self.test_f1 = BinaryAccuracy(), BinaryF1Score()
        self.save_hyperparameters()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        outputs = self.model(**xs, labels=ys)
        loss = outputs.loss.item()
        self.train_loss_list.append(loss)
        ys_ = outputs.logits.softmax(dim=1).argmax(dim=1)
        self.log('train_step_loss', loss, prog_bar=True)
        self.train_acc.update(ys_, ys)
        self.train_f1.update(ys_, ys)
        return outputs.loss

    def on_train_epoch_end(self):
        self.log_dict({
            'train_epoch_loss': np.mean(self.train_loss_list, dtype='float32')
            , 'train_acc': self.train_acc.compute()
            , 'train_f1': self.train_f1.compute()
            })
        self.train_loss_list = []
        self.train_acc.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        _, batch = batch
        xs, ys = batch
        outputs = self.model(**xs, labels=None)
        ys_ = outputs.logits.softmax(dim=1).argmax(dim=1)
        self.val_acc.update(ys_, ys)
        self.val_f1.update(ys_, ys)

    def on_validation_epoch_end(self):
        self.log_dict({
            'val_acc': self.val_acc.compute()
            , 'val_f1': self.val_f1.compute()
            })
        self.val_acc.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        _, batch = batch
        xs, ys = batch
        outputs = self.model(**xs, labels=None)
        ys_ = outputs.logits.softmax(dim=1).argmax(dim=1)
        self.test_acc.update(ys_, ys)
        self.test_f1.update(ys_, ys)

    def on_test_epoch_end(self):
        self.log_dict({
            'test_acc': self.test_acc.compute()
            , 'test_f1': self.test_f1.compute()
            })
        self.test_acc.reset()
        self.test_f1.reset()

    def predict_step(self, batch, batch_idx):
        (xs0, xs1, ys), (xs, _) = batch
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