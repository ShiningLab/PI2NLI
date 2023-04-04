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
from torchmetrics.classification import BinaryAccuracy


class LitNLIClassifier(pl.LightningModule):
    """docstring for LitNLIClassifier"""
    def __init__(self, config):
        super(LitNLIClassifier, self).__init__()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(config.LM_PATH)
        self.acc = BinaryAccuracy()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        loss = self.model(**xs, labels=ys).loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        xs0, xs1, ys = batch
        outputs0 = self.model(**xs0, labels=None)
        outputs1 = self.model(**xs1, labels=None)
        ys0_ = outputs0.logits.softmax(dim=1).argmax(dim=1)
        ys1_ = outputs1.logits.softmax(dim=1).argmax(dim=1)
        ys_ = ((ys0_ == ys1_) * (ys0_ == self.config.ENTAILMENT)).int()
        acc = self.acc(ys_, ys).item()
        self.log("val_acc", acc)

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