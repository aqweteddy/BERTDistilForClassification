import json
import os
import re

import gensim
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizerFast

class MySentences:
    def __init__(self, fname):
        self.fname = fname
        self.epoch = 0
        self.tokenizer = BertTokenizerFast.from_pretrained(
            'voidful/albert_chinese_base')

    def clean(self, text):
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                      '', text, flags=re.MULTILINE)
        # remove sent from ...
        text = text.split('--\nSent ')[0]
        # keep only eng, zh, number
        # rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
        # text = rule.sub(' ', text)
        # print(text)
        text = re.sub(" +", " ", text)
        return text

    def __iter__(self):
        cnt = 0
        print(f'Epoch: {self.epoch}')
        df = pd.read_csv(self.fname)
        print(f'Corpus - {self.fname}')
        for d in tqdm(df['text']):
            # d = self.clean(d)
            d = self.tokenizer.tokenize(d)
            # print(d)
            yield d
        print(cnt)
        cnt = 0
        self.epoch += 1


def train(data, size=300, window=10, min_count=5, workers=10, sg=1, negative=5, iter=10, max_vocab_size=None, output_file='word2vec_ptt_dcard_size_300_hs_1.bin'):
    model = gensim.models.Word2Vec(data, size=size, window=window, min_count=min_count, workers=workers, sg=sg,
                                   negative=negative, iter=iter, max_vocab_size=max_vocab_size)
    model.save(output_file)


if __name__ == '__main__':
    sents = MySentences('../../fetch_data/merge.csv')
    train(sents, size=250, window=10, workers=10,
          iter=5, output_file='w2v_250d_wordpiece.bin')
