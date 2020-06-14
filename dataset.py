from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import pandas as pd
import torch
import re
import gensim
from gaisTokenizer import Tokenizer
from sklearn.utils import shuffle


class PttDcardDataset(Dataset):
    label_dct = {'3C': 0, '購物': 1, '旅遊': 2, '政治時事': 3, '影劇': 4, '閒聊': 5, '美妝': 6, '食物': 7,
                 '運動健身': 8, '音樂': 9, '人際關係＆感情': 10, '其他': 11, '西斯': 12, '遊戲': 13, 'ACG': 14, '交通工具': 15}

    def __init__(self, file, w2v_model=None, maxlen=256, mode='bert'):
        self.maxlen = maxlen
        self.tokenizer = BertTokenizerFast.from_pretrained(
            'voidful/albert_chinese_base')
        self.mode = mode
        if self.mode == 'lstm':
            self.w2id = {
                w: w2v_model.wv.vocab[w].index for w in w2v_model.wv.vocab} if w2v_model else None
            self.data, self.labels = self.load_data(file, mode='lstm')
        elif self.mode == 'bert':
            self.data, self.labels = self.load_data(file, mode='bert')

        print(f'num_data: {len(self.data)}')

        # label_dct = set(self.labels)
        # self.label_dct = {k: idx for idx, k in enumerate(label_dct)}
        # print(self.label_dct)

    @staticmethod
    def load_data(file, mode='bert'):
        df = pd.read_json(file)
        df = shuffle(df)
        label = df['category'].tolist()
        if mode == 'bert':
            text = df['text'].tolist()
        elif mode == 'lstm':
            text = df['seg_text'].to_list()

        return text, label

    def preprocess_text(self, text: str):
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                      '', text, flags=re.MULTILINE)
        # remove sent from ...
        text = text.split('--\nSent ')[0]
        # keep only eng, zh, number
        # rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
        # text = rule.sub(' ', text)
        # print(text)
        text = re.sub(" +", " ", text)

        if len(text) > self.maxlen:
            text = text[len(text) // 4:]

        return text

    def __len__(self):
        return len(self.data)

    def get_bert(self, idx):
        text = self.preprocess_text(self.data[idx])

        inp = self.tokenizer.encode_plus(
            text=text, add_special_tokens=True, max_length=self.maxlen)
        inp_ids, type_ids = inp['input_ids'], inp['token_type_ids']
        attn_mask = inp['attention_mask']
        padding_length = self.maxlen - len(inp_ids)
        inp_ids = inp_ids + ([0] * padding_length)
        attn_mask = attn_mask + ([0] * padding_length)
        type_ids = type_ids + ([0] * padding_length)

        return (torch.tensor(inp_ids),
                torch.tensor(type_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.label_dct[self.labels[idx]]))

    def get_lstm(self, idx):
        tokens = self.data[idx]
        tokens = [self.w2id[token.strip()]
                  for token in tokens if token in self.w2id]
        # tokens = [self.w2v.wv.vocab[token].index if token in w2v.wv else 0 for token in tokens]

        tokens = tokens[:self.maxlen]
        tokens = tokens + [0] * (self.maxlen - len(tokens))
        return torch.tensor(tokens), torch.tensor(self.label_dct[self.labels[idx]])

    def __getitem__(self, idx):
        if self.mode == 'bert':
            return self.get_bert(idx)
        elif self.mode == 'lstm':
            return self.get_lstm(idx)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    w2v = gensim.models.Word2Vec.load('word2vec/w2v_250d_wordpiece.bin')
    ds = PttDcardDataset('../fetch_data/merge_train.csv', w2v, maxlen=128)
    print(ds[3])
    # from torch.utils.data import DataLoader
    for batch in DataLoader(ds, batch_size=32, num_workers=10):
        print(batch)
