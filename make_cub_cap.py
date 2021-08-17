import os

import numpy as np
import torch
# from gensim.models import FastText, Word2Vec
from gensim.models import FastText
from nltk.tokenize import sent_tokenize, word_tokenize


def preprocess(list_of_words, seq_len=32, sym_pad='<pad>', sym_eos='<eos>'):
    processed_list_of_words = []
    for words in list_of_words:
        words_trunc = words[:seq_len - 1]
        words_trunc += [sym_eos]
        word_len = len(words_trunc)
        if seq_len > word_len:
            words_trunc.extend([sym_pad] * (seq_len - word_len))

        processed_list_of_words.append(words_trunc)
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
    emb_size = 128
    len_window = 3
    epochs = 10
    model_name = 'fasttext'
    sym_pad = '<pad>'
    sym_eos = '<eos>'

    # train set
    with open(os.path.join(dataroot, 'cub/text_trainvalclasses.txt'), 'r') as file:
        sentences = sent_tokenize(file.read())

    list_of_words = [word_tokenize(s) for s in sentences]
    list_of_words = preprocess(list_of_words, seq_len=seq_len, sym_pad=sym_pad, sym_eos=sym_eos)

    model = FastText(size=emb_size, window=len_window, min_count=min_count)
    model.build_vocab(sentences=list_of_words)
    model.train(sentences=list_of_words, total_examples=len(list_of_words), epochs=epochs)

    dataset_train = convert(list_of_words, model, seq_len, emb_size)

    # test set
    with open(os.path.join(dataroot, 'cub/text_testclasses.txt'), 'r') as file:
        sentences = sent_tokenize(file.read())

    list_of_words = [word_tokenize(s) for s in sentences]
    list_of_words = preprocess(list_of_words, seq_len=seq_len, sym_pad=sym_pad, sym_eos=sym_eos)
    dataset_test = convert(list_of_words, model, seq_len, emb_size)

    dataset_train = torch.from_numpy(dataset_train)
    dataset_test = torch.from_numpy(dataset_test)

    train_min = dataset_train.min()
    train_max = dataset_train.max()
    train_mean = dataset_train.mean()
    train_std = dataset_train.std()

    print(dataset_train.size())
    print(dataset_test.size())
    print(train_min.item(), train_max.item())
    print(train_mean.size(), train_std.size())

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

    model.save(os.path.join(dataroot, 'cub/processed/{}.model'.format(model_name)))


if __name__ == '__main__':
    main()
