import os

import numpy as np
import torch
from gensim.models import Word2Vec
# from gensim.models import FastText
from nltk.tokenize import sent_tokenize, word_tokenize

from utils import OrderedCounter


def preprocess(list_of_words, vocab, seq_len=32, sym_exc='<exc>', sym_pad='<pad>', sym_eos='<eos>'):
    processed_list_of_words = []
    for words in list_of_words:
        words_trunc = words[:seq_len - 1]
        words_trunc += [sym_eos]
        word_len = len(words_trunc)
        if seq_len > word_len:
            words_trunc.extend([sym_pad] * (seq_len - word_len))

        processed_list_of_words.append(list(map(lambda w: w if w in vocab else sym_exc, words_trunc)))
    return processed_list_of_words


def convert(texts, model, seq_len, emb_size):
    dataset = np.zeros(shape=(len(texts), seq_len, emb_size), dtype='float32')
    for i, words in enumerate(texts):
        for j, w in enumerate(words[:seq_len]):
            dataset[i][j] = model.wv.get_vector(w)

    dataset = dataset.reshape((-1, 1, seq_len, emb_size))
    return dataset


def main():
    dataroot = '/tmp/data'
    min_count = 3
    seq_len = 32
    emb_size = 64
    len_window = 3
    epochs = 10

    sym_exc = '<exc>'
    sym_pad = '<pad>'
    sym_eos = '<eos>'

    # train set
    with open(os.path.join(dataroot, 'cub/text_trainvalclasses.txt'), 'r') as file:
        sentences = sent_tokenize(file.read())
        list_of_words_train = [word_tokenize(s) for s in sentences]

    occ_register = OrderedCounter()
    for words in list_of_words_train:
        occ_register.update(words)

    vocab = [sym_exc, sym_pad, sym_eos]
    for word, occ in occ_register.items():
        if occ > min_count:
            vocab.append(word)
    list_of_words_train = preprocess(list_of_words_train, vocab,
                                     seq_len=seq_len, sym_exc=sym_exc, sym_pad=sym_pad, sym_eos=sym_eos)

    model = Word2Vec(size=emb_size, window=len_window, min_count=1)
    model.build_vocab(sentences=list_of_words_train)
    model.train(sentences=list_of_words_train, total_examples=len(list_of_words_train), epochs=epochs)
    print('vocabulary length:', len(vocab), ', model vocabulary length:', len(model.wv.vocab))

    dataset_train = convert(list_of_words_train, model, seq_len, emb_size)

    # test set
    with open(os.path.join(dataroot, 'cub/text_testclasses.txt'), 'r') as file:
        sentences = sent_tokenize(file.read())
        list_of_words_test = [word_tokenize(s) for s in sentences]

    list_of_words_test = preprocess(list_of_words_test, vocab,
                                    seq_len=seq_len, sym_exc=sym_exc, sym_pad=sym_pad, sym_eos=sym_eos)
    dataset_test = convert(list_of_words_test, model, seq_len, emb_size)

    dataset_train = torch.from_numpy(dataset_train)
    dataset_test = torch.from_numpy(dataset_test)

    train_min = dataset_train.min()
    train_max = dataset_train.max()
    train_mean = dataset_train.mean()
    train_std = dataset_train.std()

    print(dataset_train.size())
    print(dataset_test.size())
    print(train_min.item(), train_max.item())
    print(train_mean.item(), train_std.item())

    os.makedirs(os.path.join(dataroot, 'cub/processed/'), exist_ok=True)
    torch.save({
        'data': dataset_train,
        'min': train_min,
        'max': train_max,
        'mean': train_mean,
        'std': train_std
    }, os.path.join(dataroot, 'cub/processed/cub-cap-train.pt'))

    torch.save({
        'data': dataset_test,
        'min': train_min,
        'max': train_max,
        'mean': train_mean,
        'std': train_std,
    }, os.path.join(dataroot, 'cub/processed/cub-cap-test.pt'))

    model.save(os.path.join(dataroot, 'cub/processed/saved_model.model'))


if __name__ == '__main__':
    main()
