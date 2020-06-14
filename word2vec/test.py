import gensim


model = gensim.models.Word2Vec.load("word2vec_ptt_dcard_size_300_hs_1.bin")

for word in ['八卦', '我們', '女人', '女神']:
    if word in model.wv.vocab:
        print(word, [w for w, c in model.wv.similar_by_word(word=word)])
    else:
        print(word, "not in vocab")

while 1:
    word = input('input word: ')
    if word in model.wv.vocab:
        print(word, [w for w, c in model.wv.similar_by_word(word=word)])
    else:
        print(word, "not in vocab")
