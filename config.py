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
    'name': 'albert',
    'batch_size': 8,
    'weight_decay': 1e-6,
    'lr': 1e-5,
    'data': {
        'maxlen': 400
    },
    'bert_model': {
        # 'pretrained_weight': False,
        # 'vocab_size': 21128,
        'num_classes': 16
    }
}

distil_hparams = {
    'name': 'distil',
    'batch_size': 32,
    'weight_decay': 0,
    'loss_a': 0.05,
    'lr': 0.01,
    'data': {
        'maxlen': 400
    },
    'lstm_model': {
        'pretrained_weight': False,
        'vocab_size': 21128,
        'embed_size': 128,
        'hid_size': 256,
        'num_layers': 3,
        'dropout': 0.2,
        'with_attn': False,
        'num_classes': 16,
        'bert_embedding': True
    },
    'bert_model': {
        # 'pretrained_weight': False,
        # 'vocab_size': 21128,
        'num_classes': 16,
        'ckpt': 'tb_logs/albert/version_3/checkpoints/epoch=3.ckpt'
    }
}
