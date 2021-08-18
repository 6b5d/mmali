import os

import gensim.models
import numpy as np
import torch
import torch.utils.data as data


def rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=30, use_all=False):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 = [], []
    for l in l1.unique():  # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        if use_all:
            n = max(l_idx1.size(0), l_idx2.size(0))
            for _ in range(dm):
                _idx1.append(l_idx1[torch.randperm(n) % l_idx1.size(0)])
                _idx2.append(l_idx2[torch.randperm(n) % l_idx2.size(0)])
        else:
            n = min(l_idx1.size(0), l_idx2.size(0), max_d)
            l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
            for _ in range(dm):
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


class PairedMNISTSVHN(data.Dataset):
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


class PairedMNISTSVHN2(data.Dataset):
    def __init__(self, mnist, svhn, max_d=10000, dm=30, use_all=False, label=False, order='ms'):
        super().__init__()
        assert order in ['ms', 'sm']

        self.mnist = mnist
        self.svhn = svhn
        self.label = label
        self.order = order

        mnist_l, mnist_li = self.mnist.targets.sort()
        svhn_l, svhn_li = torch.from_numpy(self.svhn.labels).sort()
        idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm, use_all=use_all)
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
        idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm, use_all=use_all)
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
    def __init__(self):
        super(CUBCaption, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class CUBCaptionVector(data.Dataset):
    def __init__(self, root, split='train', normalization=None, transform=None):
        super().__init__()
        assert split in ['train', 'test']

        self.root = root
        self.split = split
        self.normalization = normalization
        self.transform = transform

        self.model = gensim.models.FastText.load(os.path.join(self.root, 'cub/processed/fasttext.model'))

        d = torch.load(os.path.join(self.root,
                                    'cub/processed/cub-cap-{}.pt'.format(split)), map_location='cpu')
        self.data = d['data']
        self.normalizer = None

        if self.normalization:
            assert normalization in ['min-max', 'mean-std']
            if self.normalization == 'min-max':
                self.normalizer = d['min'], d['max'] - d['min']
            elif self.normalization == 'mean-std':
                self.normalizer = d['mean'], d['std']

            self.data = (self.data - self.normalizer[0]) / self.normalizer[1]

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
        if self.normalization:
            embeddings = (embeddings - self.normalizer[0]) / self.normalizer[1]
        return embeddings

    def decode(self, x):
        # N, 1, H, W
        x = x.squeeze(dim=1)
        if self.normalization:
            x = self.normalizer[1] * x + self.normalizer[0]

        x = x.cpu().numpy()
        sentences = []
        for sent in x:
            words = [self.model.wv.similar_by_vector(word)[0][0] for word in sent]
            sentences.append(words)

        return sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.data[index]), index

        # index as label
        return self.data[index], index


class CUBImageFeature(data.Dataset):
    def __init__(self, root, split='train', normalization=None, transform=None):
        super().__init__()

        assert split in ['train', 'test']

        self.root = root
        self.split = split
        self.normalization = normalization
        self.transform = transform

        d = torch.load(os.path.join(self.root,
                                    'cub/processed/cub-img-{}.pt'.format(split)), map_location='cpu')
        self.data = d['data']
        self.normalizer = None

        if self.normalization:
            assert normalization in ['min-max', 'mean-std']
            if self.normalization == 'min-max':
                self.normalizer = d['min'], d['max'] - d['min']
            elif self.normalization == 'mean-std':
                self.normalizer = d['mean'], d['std']

            self.data = (self.data - self.normalizer[0]) / self.normalizer[1]

        print('shape:', self.data.size())
        print('min:', self.data.min().item(), 'max:', self.data.max().item(),
              'mean:', self.data.mean().item(), 'std:', self.data.std().item())

    def __len__(self):
        return len(self.data)

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


if __name__ == '__main__':
    import torch.utils.data
    import torchvision.utils
    import torchvision

    mnist = torchvision.datasets.MNIST('/tmp/data', train=True, transform=torchvision.transforms.ToTensor())
    loader = torch.utils.data.DataLoader(mnist, batch_size=100, shuffle=False)

    mnist_l, mnist_li = mnist.targets.sort()
    idx = mnist_li[mnist_l == 5][:100]

    tx = torchvision.transforms.ToTensor()
    x = torch.index_select(mnist.data, dim=0, index=idx).unsqueeze(dim=1) / 255.
    torchvision.utils.save_image(x, 'mnist.png', nrow=10)
