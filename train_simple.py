import pytorch_lightning as pl
from dataset import PttDcardDataset
from torch.utils import data
import random
import torch
import torch.nn as nn
from model.lstm import LstmClassifier
from gensim.models import Word2Vec


class SimpleTrainer(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(SimpleTrainer, self).__init__(*args, **kwargs)
        self.hparams = hparams
        self.w2v = Word2Vec.load(
            './word2vec/word2vec_ptt_dcard_size_250_hs_1.bin')

        self.model = LstmClassifier(
            hparams['lstm_model'], params=torch.tensor(self.w2v.wv.vectors))
        # self.model.set_embedding(torch.tensor(self.w2v.wv.vectors))

    def forward(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(
        ), lr=self.hparams['lr'], weight_decay=self.hparams['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.9)
        # optimizer = torch.optim.Adagrad(self.model.parameters())
        return {'optimizer': optimizer, 'scheduler': scheduler}

    def prepare_data(self):
        ds = PttDcardDataset('../fetch_data/seg.json',
                             w2v_model=self.w2v,
                             maxlen=self.hparams['data']['maxlen'],
                             mode='lstm'
                             )

        n_train = int(len(ds) * 0.4)
        n_dev = 10000
        self.train_set, self.dev_set, _ = data.random_split(
            ds, [n_train, n_dev, len(ds) - n_train - n_dev])

    def train_dataloader(self):
        return data.DataLoader(self.train_set,
                               batch_size=self.hparams['batch_size'],
                               shuffle=True,
                               num_workers=5,
                               drop_last=True)

    def val_dataloader(self):
        return data.DataLoader(self.dev_set,
                               batch_size=self.hparams['batch_size'],
                               #    shuffle=True,
                               num_workers=5,
                               drop_last=True)

    def training_step(self, batch, batch_idx):
        inp_ids, labels = batch
        loss, logits = self.model(inp_ids, labels=labels)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'train_mean_loss': loss_mean.item()}
        self.logger.log_metrics(logs, self.current_epoch)
        results = {'progress_bar': logs, 'train_loss': loss_mean}
        print('')
        return results

    def validation_step(self, batch, batch_idx):
        inp_ids, labels = batch
        with torch.no_grad():
            loss, logits = self.model(inp_ids, labels=labels)
        correct = (logits.max(1)[1] == labels).detach().cpu().type(torch.float)

        return {'val_loss': loss, 'correct': correct}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.stack([x['correct'] for x in outputs]).mean()
        logs = {'val_acc': acc.item(), 'val_loss': val_loss_mean}
        self.logger.log_metrics(logs, self.current_epoch)
        # self.logger.log_hyperparams(self.hparams, logs)
        return {'val_loss': val_loss_mean.cpu(), 'progress_bar': logs}
