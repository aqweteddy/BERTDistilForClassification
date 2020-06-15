import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
# from model.lstm import LstmClassifier
from gensim.models import Word2Vec
from torch.utils import data
from transformers import (AlbertForSequenceClassification,
                          BertForSequenceClassification)

from dataset import PttDcardDataset


class BertTrainer(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(BertTrainer, self).__init__(*args, **kwargs)
        self.hparams = hparams
        # self.model = LstmClassifier(hparams['lstm_model'])
        self.model = BertForSequenceClassification.from_pretrained(
            'hfl/chinese-roberta-wwm-ext', num_labels=hparams['bert_model']['num_classes'])
        
    def forward(self, inp_ids, attn_masks, type_ids):
        with torch.no_grad():
            bert_logits = self.model(
                input_ids=inp_ids, attention_mask=attn_masks, token_type_ids=type_ids)[0]
        return bert_logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(
        ), lr=self.hparams['lr'], weight_decay=self.hparams['weight_decay'])
        return optimizer

    def prepare_data(self):
        ds = PttDcardDataset('./data/seg.json',
                             w2v_model=None,
                             maxlen=self.hparams['data']['maxlen'],
                             mode='bert'
                             )
        n_train = int(0.65 * len(ds))
        n_dev = 10000
        print(f'num_train: {n_train}')
        self.train_set, self.dev_set, _ = data.random_split(
            ds, [n_train, n_dev, len(ds) - n_train - n_dev])

    def train_dataloader(self):
        return data.DataLoader(self.train_set,
                               batch_size=self.hparams['batch_size'],
                               shuffle=True,
                               num_workers=8,
                               drop_last=True)

    def val_dataloader(self):
        return data.DataLoader(self.dev_set,
                               batch_size=self.hparams['batch_size'],
                               num_workers=5,
                               drop_last=True)


    def training_step(self, batch, batch_idx):
        inp_ids, type_ids, attn_masks, labels = batch
        loss, logits = self.model(
            input_ids=inp_ids, attention_mask=attn_masks, token_type_ids=type_ids, labels=labels)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'train_mean_loss': loss_mean.item()}
        self.logger.log_metrics(logs, self.current_epoch)
        results = {'progress_bar': logs, 'train_loss': loss_mean}
        return results

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            inp_ids, type_ids, attn_masks, labels = batch
            loss, logits = self.model(
                input_ids=inp_ids, attention_mask=attn_masks, token_type_ids=type_ids, labels=labels)
        correct = (logits.max(1)[1] == labels).detach().cpu().type(torch.float)

        return {'val_loss': loss, 'correct': correct}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.stack([x['correct'] for x in outputs]).mean()
        logs = {'val_acc': acc.item(), 'val_loss': val_loss_mean}
        self.logger.log_metrics(logs, self.current_epoch)
        # self.logger.log_hyperparams(self.hparams, logs)
        return {'val_loss': val_loss_mean.cpu(), 'progress_bar': logs}
