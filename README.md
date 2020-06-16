# An Unofficial Implementation for Distilling Task-Specific Knowledge from BERT into Simple Neural Networks

* Document Classification

## Dataset

* dcard + ptt 共 20 萬篇文章

## Environement

* Python 3.7
* `pip install -r requirements.txt`

## training

### 1. Train gensim word2vec

* see `word2vec/train.py`

### 2. 更改資料集路徑、欄位等

* see `dataset.py`

### 3. set config

* See `config.py`

### 4. train bert model

* default `RoBERTa`
* `python main.py --model=bert`

### 5. train distilled LSTM

* set config in `config.py` distil_hparams
* `python main.py --model=bert`

## 結果

### Roberta + LSTM 3 layers

* Roberta 3 epoch acc. : 90 %
* Distill 3 epoch LSTM : 

```js
'data': {
    'maxlen': 350
},
'lstm_model': {
    'freeze': False,
    'embed_size': 250,
    'hid_size': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'with_attn': False,
    'num_classes': 16,
},
'bert_model': {
    'num_classes': 16,
    'ckpt': 'logs/roberta/version_6/epoch=1.ckpt'
}
```