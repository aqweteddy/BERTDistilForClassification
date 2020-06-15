import pytorch_lightning as pl
from dataset import PttDcardDataset, ConcatDataset
from torch.utils import data
import random
import torch
import torch.nn as nn
from model.lstm import LstmClassifier
from gensim.models import Word2Vec
from transformers import BertForSequenceClassification
from train_bert import BertTrainer


class WeightedLoss(nn.Module):
    def __init__(self, a: int=0.5):
        super(WeightedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.a = a

    def forward(self,lstm_logits, bert_logits, labels):
        """Weighted MSE

        Args:
            lstm_logits (tensor): batch, num_class
            bert_logits (tensor): batch, num_class
            labels (tesnor): batch

        Returns:
            tensor: loss
        """
        # print(lstm_logits.shape, bert_logits.shape, labels.shape)
        # lstm_logits = torch.softmax(lstm_logits, dim=1)
        # bert_logits = torch.softmax(bert_logits, dim=1)
        return self.a * self.ce(lstm_logits, labels) + (1. - self.a) * self.mse(lstm_logits, bert_logits)


class DistilTrainer(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(DistilTrainer, self).__init__(*args, **kwargs)
        self.hparams = hparams
        self.bert = BertTrainer.load_from_checkpoint(hparams['bert_model']['ckpt'])
        self.w2v = Word2Vec.load('./word2vec/word2vec_ptt_dcard_size_250_hs_1.bin')
        self.lstm = LstmClassifier(hparams['lstm_model'], params=torch.tensor(self.w2v.wv.vectors))
        # self.bert = AlbertForSequenceClassification.from_pretrained(
        #     'voidful/albert_chinese_base', num_labels=hparams['bert_model']['num_classes'])
        self.criterion = WeightedLoss(a=hparams['loss_a'])

    def forward(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        # optimizer = torch.optim.Adagrad(self.lstm.parameters())
        optimizer = torch.optim.Adam(self.lstm.parameters(
        ), lr=self.hparams['lr'], weight_decay=self.hparams['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        return {'optimizer':optimizer, 'scheduler':scheduler}

    def prepare_data(self):
        # self.w2v = Word2Vec.load('./word2vec/word2vec_ptt_dcard_size_250_hs_1.bin')
        ds_bert = PttDcardDataset('./data/seg.json',
                             maxlen=self.hparams['data']['maxlen'],
                             mode='bert'
                             )
        ds_lstm = PttDcardDataset('./data/seg.json',
                             w2v_model=self.w2v,
                             maxlen=self.hparams['data']['maxlen'],
                             mode='lstm'
                             )
        ds_concat = ConcatDataset(ds_bert, ds_lstm)
        n_train = int(len(ds_concat) * 0.4)
        n_dev = 10000
        self.train_set, self.dev_set, _ = data.random_split(
            ds_concat, [n_train, n_dev, len(ds_concat) - n_train - n_dev])

    def train_dataloader(self):
        return data.DataLoader(self.train_set,
                               batch_size=self.hparams['batch_size'],
                               shuffle=True,
                               num_workers=5,
                               drop_last=True)

    def val_dataloader(self):
        return data.DataLoader(self.dev_set,
                               batch_size=self.hparams['batch_size'],
                               num_workers=5,
                               drop_last=True)

    def training_step(self, batch, batch_idx):
        batch_bert, batch_lstm = batch
        inp_ids, type_ids, attn_masks, labels = batch_bert
        with torch.no_grad():
            bert_logits = self.bert(
                inp_ids, attn_masks, type_ids)
        lstm_inp_ids, lstm_labels = batch_lstm
        assert (labels == lstm_labels).all()
        lstm_logits = self.lstm(lstm_inp_ids, lstm_labels)
        loss = self.criterion(lstm_logits, bert_logits, labels)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'train_mean_loss': loss_mean.item()}
        self.logger.log_metrics(logs, self.current_epoch)
        results = {'progress_bar': logs, 'train_loss': loss_mean}
        return results

    def validation_step(self, batch, batch_idx):
        batch_bert, batch_lstm = batch
        inp_ids, type_ids, attn_masks, labels = batch_bert

        with torch.no_grad():
            bert_logits = self.bert(
                inp_ids, attn_masks, type_ids)
            inp_ids, labels = batch_lstm
            lstm_logits = self.lstm(inp_ids, attn_masks)
        loss = nn.functional.cross_entropy(lstm_logits, labels)
        correct = (lstm_logits.max(1)[1] == labels).detach().cpu().type(torch.float)

        return {'val_loss': loss, 'correct': correct}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.stack([x['correct'] for x in outputs]).mean()
        logs = {'val_acc': acc.item(), 'val_loss': val_loss_mean}
        self.logger.log_metrics(logs, self.current_epoch)
        # self.logger.log_hyperparams(self.hparams, logs)
        return {'val_loss': val_loss_mean.cpu(), 'progress_bar': logs}
