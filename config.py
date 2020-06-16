lstm_hparams = {
    'name': 'LSTM',
    'batch_size': 64,
    'weight_decay': 1e-4,
    # 'lr': 5e-3,
    'lr': 0.01, # original: 0.01
    'data': {
        'maxlen': 400
    },
    'lstm_model': {
        'vocab_size': 115963,
        'embed_size': 250,
        'hid_size': 256,
        'num_layers': 2,
        'with_attn': False,
        'dropout': 0.3,
        'num_classes': 16
    }
}

bert_hparams = {
    'name': 'roberta',
    'batch_size': 12,
    'weight_decay': 0,
    'lr': 3e-5,
    'data': {
        'maxlen': 350
    },
    'bert_model': {
        # 'pretrained_weight': False,
        # 'vocab_size': 21128,
        'num_classes': 16
    }
}

distil_hparams = {
    'name': 'distil_roberta_lstm',
    'description': 'MSE(logits)',
    'batch_size': 64,
    'weight_decay': 0,
    'loss_a': 0.,
    # 'lr': 0.003,
    'lr': 0.01,
    'data': {
        'maxlen': 350
    },
    'lstm_model': {
        'freeze': False,
        # 'vocab_size': 21128,
        'embed_size': 250,
        'hid_size': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'with_attn': False,
        'num_classes': 16,
    },
    'bert_model': {
        # 'pretrained_weight': False,
        # 'vocab_size': 21128,
        'num_classes': 16,
        'ckpt': 'logs/roberta/version_6/epoch=1.ckpt'
    }
}
