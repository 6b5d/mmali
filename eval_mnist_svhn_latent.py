import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision

import models.mnist
import models.svhn
import options

opt = options.parser.parse_args()
print(opt)

mnist_img_shape = (1, 28, 28)
svhn_channels = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(classifier, dataloader_train, optimizer):
    classifier.train()
    running_loss = 0.
    for i, (x, y) in enumerate(dataloader_train):
        x = x.to(device)
        y = y.to(device)

        y_pred = classifier(x)
        loss = F.cross_entropy(y_pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader_train)


@torch.no_grad()
def test(classifier, dataloader_test):
    classifier.eval()
    total = 0
    correct = 0
    for i, (x, y) in enumerate(dataloader_test):
        x = x.to(device)
        y = y.to(device)

        y_pred = classifier(x)
        _, predicted = torch.max(y_pred, dim=1)

        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    return correct, total


def main():
    if opt.dataset == 'mnist':
        dataset_train = torchvision.datasets.MNIST(opt.dataroot,
                                                   train=True,
                                                   download=True,
                                                   transform=torchvision.transforms.ToTensor())
        dataset_test = torchvision.datasets.MNIST(opt.dataroot,
                                                  train=False,
                                                  download=True,
                                                  transform=torchvision.transforms.ToTensor())
        encoder = models.mnist.Encoder(img_shape=mnist_img_shape,
                                       latent_dim=opt.latent_dim if opt.deterministic else 2 * opt.latent_dim)
    elif opt.dataset == 'svhn':
        dataset_train = torchvision.datasets.SVHN(opt.dataroot,
                                                  split='train',
                                                  download=True,
                                                  transform=torchvision.transforms.ToTensor())
        dataset_test = torchvision.datasets.SVHN(opt.dataroot,
                                                 split='test',
                                                 download=True,
                                                 transform=torchvision.transforms.ToTensor())
        encoder = models.svhn.Encoder(channels=svhn_channels,
                                      latent_dim=opt.latent_dim if opt.deterministic else 2 * opt.latent_dim)
    else:
        raise NotImplementedError

    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.n_cpu,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=False)

    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=opt.batch_size,
                                                  num_workers=opt.n_cpu,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  drop_last=False)

    conditional = models.DeterministicConditional if opt.deterministic else models.GaussianConditional
    encoder = conditional(encoder)
    state_dict = torch.load(opt.checkpoint, map_location=device)
    encoder.load_state_dict(state_dict)

    input_dim = opt.latent_dim
    if opt.content_only:
        input_dim = opt.latent_dim - opt.style_dim
        slicer = (slice(None), slice(opt.style_dim, None))  # [:, style_dim:]
        encoder = models.SliceLayer(encoder, slicer)
    elif opt.style_only:
        input_dim = opt.style_dim
        slicer = (slice(None), slice(None, opt.style_dim))
        encoder = models.SliceLayer(encoder, slicer)  # [:, :opt.style_dim]

    encoder.requires_grad_(False)
    classifier = models.LinearClassifier(encoder, input_dim, 10)
    classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    acc_max = 0.0
    for epoch in range(opt.n_epochs):
        running_loss = train(classifier, dataloader_train, optimizer)
        correct, total = test(classifier, dataloader_test)

        acc = correct / total * 100
        if acc > acc_max:
            acc_max = acc
        print(
            '[Epoch {:d}/{:d}] [Train]'
            ' [loss: {:f}]'.format(epoch, opt.n_epochs, running_loss)
        )
        print(
            '[Epoch {:d}/{:d}] [Test]'
            ' [Examples: {:d}]'
            ' [Accuracy: {:f}]'
            ' [Max Accuracy: {:f}]'.format(epoch, opt.n_epochs, total, acc, acc_max)
        )

    print('{:.6f}'.format(acc_max))


if __name__ == '__main__':
    main()
