import json
import os
import pickle

import gensim.models
import numpy as np
import torch
import torch.utils.data as data
from nltk.tokenize import sent_tokenize, word_tokenize

from utils import OrderedCounter


def rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, data_multiplication=30, use_all=False, percentage=1.0):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 = [], []
    for l in l1.unique():  # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        if use_all:
            # n = max(l_idx1.size(0), l_idx2.size(0))
            n = min(int(l_idx1.size(0) * percentage), int(l_idx2.size(0) * percentage))
            for _ in range(data_multiplication):
                _idx1.append(l_idx1[torch.randperm(n)])
                _idx2.append(l_idx2[torch.randperm(n)])
        else:
            n = min(l_idx1.size(0), l_idx2.size(0), max_d)
            l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
            for _ in range(data_multiplication):
                _idx1.append(l_idx1[torch.randperm(n)])
                _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)


def InfiniteSampler(n):
    i = 0
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


class PairedMNISTSVHNOld(data.Dataset):
    def __init__(self, root, mnist, svhn, train=True, label=False):
        super().__init__()
        self.root = root
        self.mnist = mnist
        self.svhn = svhn
        self.label = label

        if train:
            self.mnist_idx = torch.load(os.path.join(self.root, 'train-ms-mnist-idx.pt'))
            self.svhn_idx = torch.load(os.path.join(self.root, 'train-ms-svhn-idx.pt'))
        else:
            self.mnist_idx = torch.load(os.path.join(self.root, 'test-ms-mnist-idx.pt'))
            self.svhn_idx = torch.load(os.path.join(self.root, 'test-ms-svhn-idx.pt'))

        assert len(self.mnist_idx) == len(self.svhn_idx)
        print('total:', len(self.mnist_idx))

    def __len__(self):
        return len(self.mnist_idx)

    def __getitem__(self, index):
        # assert self.mnist[self.mnist_idx[index]][1] == self.svhn[self.svhn_idx[index]][1]
        # if self.label:
        #     return self.mnist[self.mnist_idx[index]][0], self.svhn[self.svhn_idx[index]][0], \
        #            self.mnist[self.mnist_idx[index]][1]
        return self.mnist[self.mnist_idx[index]][0], self.svhn[self.svhn_idx[index]][0]


class PairedMNISTSVHN(data.Dataset):
    def __init__(self, mnist, svhn, max_d=10000, dm=30, use_all=False, percentage=1.0, label=False, order='ms'):
        super().__init__()
        assert order in ['ms', 'sm']

        self.mnist = mnist
        self.svhn = svhn
        self.label = label
        self.order = order

        mnist_l, mnist_li = self.mnist.targets.sort()
        svhn_l, svhn_li = torch.from_numpy(self.svhn.labels).sort()
        idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, data_multiplication=dm,
                                       use_all=use_all, percentage=percentage)
        self.mnist_idx = idx1
        self.svhn_idx = idx2

        assert len(self.mnist_idx) == len(self.svhn_idx)
        print('total: {}'.format(len(self.mnist_idx)))

    def __len__(self):
        return len(self.mnist_idx)

    def __getitem__(self, index):
        if self.order == 'ms':
            if self.label:
                return self.mnist[self.mnist_idx[index]][0], self.svhn[self.svhn_idx[index]][0], \
                       self.mnist[self.mnist_idx[index]][1]

            return self.mnist[self.mnist_idx[index]][0], self.svhn[self.svhn_idx[index]][0]
        elif self.order == 'sm':
            if self.label:
                return self.svhn[self.svhn_idx[index]][0], self.mnist[self.mnist_idx[index]][0], \
                       self.mnist[self.mnist_idx[index]][1]

            return self.svhn[self.svhn_idx[index]][0], self.mnist[self.mnist_idx[index]][0]


class MultiMNIST(data.Dataset):
    def __init__(self, mnists, max_d=10000, dm=30):
        super().__init__()
        self.mnists = mnists
        self.max_d = max_d
        self.dm = dm
        self.n_sets = len(mnists)

        mnist_l, mnist_li = self.mnists[0].targets.sort()
        self.mnist_indexes = self.rand_match_on_idx(mnist_l, mnist_li)

    def rand_match_on_idx(self, label, label_index):
        indexes = [[] for _ in range(self.n_sets)]
        for l in label.unique():
            l_idx = label_index[label == l]
            n = min(self.max_d, l_idx.size(0))

            for _ in range(self.dm):
                for idx in indexes:
                    idx.append(l_idx[torch.randperm(n)])

        return [torch.cat(idx) for idx in indexes]

    def __len__(self):
        return len(self.mnist_indexes[0])

    def __getitem__(self, index):
        imgs = []
        for i in range(self.n_sets):
            dset = self.mnists[i]
            idx = self.mnist_indexes[i]
            img = dset[idx[index]][0]
            imgs.append(img)
        return imgs


class MNISTSVHN(data.Dataset):
    def __init__(self, mnist, svhn, max_d=10000, dm=30, use_all=False, label=False, order='ms'):
        super().__init__()
        assert order in ['ms', 'sm']
        self.mnist = mnist
        self.svhn = svhn
        self.label = label
        self.order = order

        mnist_l, mnist_li = self.mnist.targets.sort()
        svhn_l, svhn_li = torch.from_numpy(self.svhn.labels).sort()
        idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, data_multiplication=dm,
                                       use_all=use_all)
        self.mnist_idx = idx1
        self.svhn_idx = idx2

        assert len(self.mnist_idx) == len(self.svhn_idx)
        print('total: {}'.format(len(self.mnist_idx)))

    def __len__(self):
        return len(self.mnist_idx)

    def __getitem__(self, index):
        if self.order == 'ms':
            if self.label:
                return self.mnist[self.mnist_idx[index]][0], self.svhn[self.svhn_idx[index]][0], \
                       self.mnist[self.mnist_idx[index]][1]

            return self.mnist[self.mnist_idx[index]][0], self.svhn[self.svhn_idx[index]][0]
        elif self.order == 'sm':
            if self.label:
                return self.svhn[self.svhn_idx[index]][0], self.mnist[self.mnist_idx[index]][0], \
                       self.mnist[self.mnist_idx[index]][1]

            return self.svhn[self.svhn_idx[index]][0], self.mnist[self.mnist_idx[index]][0]


class CUBCaption(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        assert split in ['train', 'test']

        self.root = root
        self.split = split
        self.transform = transform

        if self.split == 'train':
            raw_data_path = os.path.join(self.root, 'cub/text_trainvalclasses.txt')
        else:
            raw_data_path = os.path.join(self.root, 'cub/text_testclasses.txt')

        self.vocab = self.create_vocab()
        self.data = self.create_data(raw_data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.data[index]), index

        # index as label
        return self.data[index], index

    def create_vocab(self):
        if os.path.exists(os.path.join(self.root, 'cub/processed/vocab.json')):
            with open(os.path.join(self.root, 'cub/processed/vocab.json'), 'r') as file:
                vocab = json.load(file)
        else:
            with open(os.path.join(self.root, 'cub/text_trainvalclasses.txt'), 'r') as file:
                sentences = sent_tokenize(file.read())
                texts = [word_tokenize(s) for s in sentences]

            occ_register = OrderedCounter()
            w2i = dict()
            i2w = dict()

            special_tokens = ['<exc>', '<pad>', '<eos>']
            for st in special_tokens:
                i2w[len(w2i)] = st
                w2i[st] = len(w2i)

            unq_words = []

            for words in texts:
                occ_register.update(words)

            for w, occ in occ_register.items():
                if occ > 3 and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)
                else:
                    unq_words.append(w)

            assert len(w2i) == len(i2w)

            vocab = dict(w2i=w2i, i2w=i2w)
            with open(os.path.join(self.root, 'cub/processed/vocab.json'), 'w') as file:
                json.dump(vocab, file, ensure_ascii=False)

            with open(os.path.join(self.root, 'cub/processed/vocab.json'), 'r') as file:
                vocab = json.load(file)

        return vocab

    def create_data(self, raw_data_path):

        with open(raw_data_path, 'r') as file:
            sentences = sent_tokenize(file.read())
            texts = [word_tokenize(s) for s in sentences]

        dataset = []
        seq_len = 32
        for words in texts:
            words_trunc = words[:seq_len - 1]
            words_trunc += ['<eos>']
            word_len = len(words_trunc)
            if seq_len > word_len:
                words_trunc.extend(['<pad>'] * (seq_len - word_len))
            dataset.append(list(map(lambda w: self.vocab['w2i'].get(w, self.vocab['w2i']['<eos>']), words_trunc)))
        return dataset


class CUBCaptionVector(data.Dataset):
    def __init__(self, root, split='train', emb_size=128, normalization=None, margin=1.0, transform=None):
        super().__init__()
        assert split in ['train', 'test']

        self.root = root
        self.split = split
        self.emb_size = emb_size
        self.normalization = normalization
        self.margin = margin
        self.transform = transform

        self.model = gensim.models.Word2Vec.load(
            os.path.join(self.root, 'cub/processed/savedmodel{}.model'.format(emb_size)))

        d = torch.load(os.path.join(self.root,
                                    'cub/processed/cub-cap-{}{}.pt'.format(split, emb_size)), map_location='cpu')
        self.data = d['data']
        self.normalizer = None

        if self.normalization:
            assert normalization in ['min-max', 'mean-std']
            if self.normalization == 'min-max':
                self.normalizer = self.margin * 2.0 / (d['max'] - d['min']), \
                                  self.margin * (-2.0 * d['min'] / (d['max'] - d['min']) - 1.0)
            elif self.normalization == 'mean-std':
                self.normalizer = 1.0 / d['std'], -d['mean'] / d['std']

            self.data = self.data * self.normalizer[0] + self.normalizer[1]

        print('shape:', self.data.size())
        print('min:', self.data.min().item(), 'max:', self.data.max().item(),
              'mean:', self.data.mean().item(), 'std:', self.data.std().item())

    def encode(self, texts):
        embeddings = []
        for words in texts:
            emb = np.stack([self.model.wv.get_vector(w) for w in words], axis=0)
            embeddings.append(emb)

        embeddings = np.stack(embeddings, axis=0)
        embeddings = np.expand_dims(embeddings, axis=1)
        embeddings = torch.from_numpy(embeddings)
        if self.normalizer:
            embeddings = embeddings * self.normalizer[0] + self.normalizer[1]
        return embeddings

    def decode(self, x):
        # N, 1, H, W
        x = x.squeeze(dim=1)
        if self.normalizer:
            x = (x - self.normalizer[1]) / self.normalizer[0]

        x = x.cpu().numpy()
        sentences = [[
            self.model.wv.similar_by_vector(word)[0][0] for word in sent
        ] for sent in x]

        return sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.data[index]), index

        # index as label
        return self.data[index], index


class CUBCaptionFeature(data.Dataset):
    def __init__(self, root, split='train', normalization=None, margin=1.0, transform=None):
        super().__init__()
        assert split in ['train', 'test']

        self.root = root
        self.split = split
        self.normalization = normalization
        self.margin = margin
        self.transform = transform

        with open(os.path.join(self.root, 'cub/processed/{}/char-CNN-RNN-embeddings.pickle'.format(split)), 'rb') as f:
            feat = pickle.load(f, encoding='bytes')
            feat = torch.from_numpy(np.array(feat))

        self.data = feat
        self.normalizer = None
        if self.normalization:
            assert normalization in ['min-max', 'mean-std']
            if self.normalization == 'min-max':
                d_min, d_max = self.data.min(), self.data.max()
                self.normalizer = margin * (2.0 / (d_max - d_min)), \
                                  margin * (-2.0 * d_min / (d_max - d_min) - 1.0)
            elif self.normalization == 'mean-std':
                d_mean, d_std = self.data.min(), self.data.max()
                self.normalizer = 1.0 / d_std, -d_mean / d_std

            self.data = self.data * self.normalizer[0] + self.normalizer[1]

        print('shape:', self.data.size())
        print('min:', self.data.min().item(), 'max:', self.data.max().item(),
              'mean:', self.data.mean().item(), 'std:', self.data.std().item())

    def __len__(self):
        return len(self.data) * 10

    def __getitem__(self, index):
        idx1, idx2 = index // 10, index % 10

        if self.transform:
            return self.transform(self.data[idx1][idx2]), index

        # index as label
        return self.data[idx1][idx2], index


class CUBImageFeature(data.Dataset):
    def __init__(self, root, split='train', normalization=None, margin=1.0, transform=None):
        super().__init__()

        assert split in ['train', 'test']

        self.root = root
        self.split = split
        self.normalization = normalization
        self.margin = margin
        self.transform = transform

        d = torch.load(os.path.join(self.root,
                                    'cub/processed/cub-img-{}.pt'.format(split)), map_location='cpu')
        self.data = d['data']
        self.normalizer = None

        if self.normalization:
            assert normalization in ['min-max', 'mean-std']
            if self.normalization == 'min-max':
                self.normalizer = margin * (2.0 / (d['max'] - d['min'])), \
                                  margin * (-2.0 * d['min'] / (d['max'] - d['min']) - 1.0)
            elif self.normalization == 'mean-std':
                self.normalizer = 1.0 / d['std'], -d['mean'] / d['std']

            self.data = self.data * self.normalizer[0] + self.normalizer[1]

        print('shape:', self.data.size())
        print('min:', self.data.min().item(), 'max:', self.data.max().item(),
              'mean:', self.data.mean().item(), 'std:', self.data.std().item())

    def __len__(self):
        return len(self.data)

    def decode(self, x):
        if self.normalizer:
            x = (x - self.normalizer[1]) / self.normalizer[0]
        return x

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.data[index]), index

        # index as label
        return self.data[index], index


class CaptionImagePair(data.Dataset):
    def __init__(self, cap, img):
        super().__init__()
        self.cap = cap
        self.img = img

        assert len(self.cap) == 10 * len(self.img)

    def __len__(self):
        return len(self.cap)

    def __getitem__(self, index):
        return self.cap[index][0], self.img[index // 10][0]


class CUBCaptionImageFeature(data.Dataset):
    def __init__(self, root, split='train'):
        super().__init__()

        self.root = root
        self.split = split

        with open(os.path.join(self.root, 'cub/processed/{}/char-CNN-RNN-embeddings.pickle'.format(split)), 'rb') as f:
            cap_feat = pickle.load(f, encoding='bytes')
            cap_feat = torch.from_numpy(np.array(cap_feat))
        self.cap_feat = cap_feat

        d = torch.load(os.path.join(self.root, 'cub/processed/cub-img-{}.pt'.format(split)), map_location='cpu')
        self.img_feat = d['data']

    def __len__(self):
        return len(self.cap_feat) * 10

    def __getitem__(self, index):
        idx1, idx2 = index // 10, index % 10

        return self.cap_feat[idx1][idx2], self.img_feat[idx1]


if __name__ == '__main__':
    import torch.utils.data
    import torchvision.utils
    import torchvision

    x1_dataset = torchvision.datasets.MNIST('/tmp/data', train=True, download=True,
                                            transform=torchvision.transforms.ToTensor())
    x2_dataset = torchvision.datasets.MNIST('/tmp/data', train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.RandomRotation((90, 90)),
                                                torchvision.transforms.ToTensor()
                                            ]))
    x3_dataset = torchvision.datasets.MNIST('/tmp/data', train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.RandomRotation((-90, -90)),
                                                torchvision.transforms.ToTensor()
                                            ]))
    paired_dataset = MultiMNIST([x1_dataset, x2_dataset, x3_dataset], max_d=100, dm=30)

    loader = torch.utils.data.DataLoader(paired_dataset, batch_size=32, shuffle=True)

    batch = next(iter(loader))
    # print(len(batch[0]))
    x1, x2, x3 = batch
    samples = torch.cat([x1, x2, x3], dim=0)
    # print(samples.size())
    torchvision.utils.save_image(samples, 'mnist.png', nrow=32)
