import pytorch_lightning as pl
from train_simple import SimpleTrainer
from train_bert import BertTrainer
from train_distil import DistilTrainer
from config import lstm_hparams, bert_hparams,distil_hparams
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--model', type=str, help='bert lstm or joint')
parser.add_argument('--find_lr', type=int, default=False)
args = parser.parse_args()

if args.model == 'bert':
    hparams = bert_hparams
    model = BertTrainer(hparams)
elif args.model =='lstm':
    hparams = lstm_hparams
    model = SimpleTrainer(hparams)
elif args.model == 'distil':
    hparams = distil_hparams
    model = DistilTrainer(hparams)

# model = SimpleTrainer(hparams)

early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_acc',
    min_delta=0.001,
    patience=3,
    verbose=True,
    mode='max'
)

save_ckpt_callback = pl.callbacks.ModelCheckpoint(os.path.join('tb_logs', hparams['name']), save_top_k=2, mode='max', monitor='val_acc')

# model = SimpleTrainer(hparams)

trainer = pl.Trainer(logger=pl.loggers.TensorBoardLogger('tb_logs', name=hparams['name']),
                     check_val_every_n_epoch=1,
                    #  resume_from_checkpoint='tb_logs/roberta/version_0/checkpoints/epoch=1.ckpt',
                     checkpoint_callback=save_ckpt_callback,
                     gpus=1,
                     max_epochs=5,
                     )
# finder = trainer.lr_find(model)
# lr = finder.suggestion()
# print(lr)
if args.find_lr == 1:
    finder = trainer.lr_find(model)
    lr = finder.suggestion()
    print(lr)
else:
    trainer.fit(model)