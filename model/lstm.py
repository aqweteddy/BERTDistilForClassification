import torch
import torch.nn as nn


class LstmClassifier(nn.Module):
    def __init__(self, hparams, params=None):
        super(LstmClassifier, self).__init__()
        self.with_attn = hparams['with_attn']
        self.hparams = hparams
        if params is not None:
            self.embedding = nn.Embedding.from_pretrained(params, freeze=True, padding_idx=0)
        else:
            self.embedding = nn.Embedding(hparams['dct_size'], hparams['embed_size'], 0)
        self.lstm = nn.LSTM(hparams['embed_size'],
                            hparams['hid_size'],
                            num_layers=hparams['num_layers'],
                            batch_first=True,
                            bidirectional=True,
                            dropout=hparams['dropout']
                            )
        self.fc = nn.Sequential(nn.Linear(hparams['hid_size'] * 2, hparams['num_classes']),
                                nn.ReLU())
        self.criterion = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.4)
        # TODO
        # if hparams['with_attn']:
            # self.attn = nn.MultiheadAttention()

    def set_embedding(self, params):
        self.embedding = nn.Embedding.from_pretrained(params, freeze=False, padding_idx=0)

    def forward(self, inp_ids, attn_masks=None, labels=None):
        embed = self.dropout(self.embedding(inp_ids))

        outputs, (hidden, _) = self.lstm(embed)
        hidden = hidden.permute(1, 0, 2)
        # logits = self.fc(outputs[:,-1])
        hidden = hidden[:, -2:, :]
        hidden = torch.cat((hidden[:, -1, :], hidden[:, -2, :]), dim=1)
        logits = self.fc(hidden)
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        else:
            return logits
