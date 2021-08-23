import os.path

import torch.nn.functional as F
import torch.utils.data
import torchvision

import datasets
import models.mnist
import models.svhn
import options
import utils

opt = options.parser.parse_args()
print(opt)

mnist_img_shape = (1, 28, 28)
svhn_channels = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(classifier, dataloader_train, optimizer):
    classifier.train()
    loss_avg = 0.
    total = 0
    for x, y in dataloader_train:
        x = x.to(device)
        y = y.to(device)

        y_pred = classifier(x)
        loss = F.cross_entropy(y_pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()
        total += y.size(0)

    return loss_avg / total


@torch.no_grad()
def test(classifier, dataloader_test):
    classifier.eval()
    total = 0
    correct = 0
    for x, y in dataloader_test:
        x = x.to(device)
        y = y.to(device)

        y_pred = classifier(x)
        _, predicted = torch.max(y_pred, dim=1)

        correct += predicted.eq(y).sum().item()
        total += y.size(0)

    return correct, total


def train_and_save_classifier(dataset_train, dataset_test, classifier, filename):
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
    optimizer = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    acc_max = 0.0
    for epoch in range(opt.n_epochs):
        train(classifier, dataloader_train, optimizer)
        correct, total = test(classifier, dataloader_test)

        acc = correct / total
        if acc > acc_max:
            acc_max = acc
            torch.save(classifier.state_dict(), filename)
            print('saving with test accuracy: {:6f}'.format(acc_max))


def load_or_train_classifier():
    mnist_classifier = models.mnist.Classifier()
    mnist_classifier.to(device)

    svhn_classifier = models.svhn.Classifier()
    svhn_classifier.to(device)
    mnist_file = os.path.join(opt.checkpoint_dir, 'mnist_classifier.pt')
    if os.path.exists(mnist_file):
        print('loading from {}'.format(mnist_file))
        mnist_classifier.load_state_dict(torch.load(mnist_file, map_location=device))
    else:
        print('training mnist classifier from scratch')
        dataset_train = torchvision.datasets.MNIST(opt.dataroot,
                                                   train=True,
                                                   download=True,
                                                   transform=torchvision.transforms.ToTensor())
        dataset_test = torchvision.datasets.MNIST(opt.dataroot,
                                                  train=False,
                                                  download=True,
                                                  transform=torchvision.transforms.ToTensor())
        train_and_save_classifier(dataset_train, dataset_test, mnist_classifier, mnist_file)

    svhn_file = os.path.join(opt.checkpoint_dir, 'svhn_classifier.pt')
    if os.path.exists(svhn_file):
        print('loading from {}'.format(svhn_file))
        svhn_classifier.load_state_dict(torch.load(svhn_file, map_location=device))
    else:
        print('training svhn classifier from scratch')
        dataset_train = torchvision.datasets.SVHN(opt.dataroot,
                                                  split='train',
                                                  download=True,
                                                  transform=torchvision.transforms.ToTensor())
        dataset_test = torchvision.datasets.SVHN(opt.dataroot,
                                                 split='test',
                                                 download=True,
                                                 transform=torchvision.transforms.ToTensor())
        train_and_save_classifier(dataset_train, dataset_test, svhn_classifier, svhn_file)

    return mnist_classifier, svhn_classifier


def calc_cross_coherence(paired_loader,
                         mnist_encoder, mnist_decoder, mnist_classifier,
                         svhn_encoder, svhn_decoder, svhn_classifier):
    total = 0
    correct_m2s = 0
    correct_s2m = 0
    for x1, x2, y in paired_loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        dec_x2 = svhn_decoder(mnist_encoder(x1))
        pred_y = svhn_classifier(dec_x2)
        _, predicted = torch.max(pred_y, dim=1)
        correct_m2s += predicted.eq(y).sum().item()

        dec_x1 = mnist_decoder(svhn_encoder(x2))
        pred_y = mnist_classifier(dec_x1)
        _, predicted = torch.max(pred_y, dim=1)
        correct_s2m += predicted.eq(y).sum().item()

        total += y.size(0)

    return correct_m2s / total, correct_s2m / total


def calc_joint_coherence(mnist_decoder, mnist_classifier, svhn_decoder, svhn_classifier, total=10000):
    z = torch.randn(total, opt.latent_dim).to(device)

    x_mnist = mnist_decoder(z)
    x_svhn = svhn_decoder(z)

    y_mnist_pred = mnist_classifier(x_mnist)
    y_svhn_pred = svhn_classifier(x_svhn)

    _, pred_m = torch.max(y_mnist_pred, dim=1)
    _, pred_s = torch.max(y_svhn_pred, dim=1)

    correct = pred_m.eq(pred_s).sum().item()

    return correct / total


def calc_synergy_coherence(paired_loader, mnist_encoder, mnist_decoder, mnist_classifier,
                           svhn_encoder, svhn_decoder, svhn_classifier):
    total = 0
    correct_m = 0
    correct_s = 0
    for x1, x2, y in paired_loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        z1 = mnist_encoder.module(x1)
        z2 = svhn_encoder.module(x2)
        z_joint = utils.reparameterize(utils.joint_posterior(z1, z2, average=True))
        s1 = utils.reparameterize(z1)[:, :opt.style_dim]
        s2 = utils.reparameterize(z2)[:, :opt.style_dim]
        c = z_joint[:, opt.style_dim:]

        dec_x1 = mnist_decoder(torch.cat([s1, c], dim=1))
        dec_x2 = svhn_decoder(torch.cat([s2, c], dim=1))
        pred_y1 = mnist_classifier(dec_x1)
        pred_y2 = svhn_classifier(dec_x2)

        _, predicted1 = torch.max(pred_y1, dim=1)
        _, predicted2 = torch.max(pred_y2, dim=1)

        correct_m += predicted1.eq(y).sum().item()
        correct_s += predicted2.eq(y).sum().item()
        total += y.size(0)

    return correct_m / total, correct_s / total


def main():
    paired_dataset = datasets.PairedMNISTSVHN2(torchvision.datasets.MNIST(opt.dataroot,
                                                                          train=False,
                                                                          download=True,
                                                                          transform=torchvision.transforms.ToTensor()),
                                               torchvision.datasets.SVHN(opt.dataroot,
                                                                         split='test',
                                                                         download=True,
                                                                         transform=torchvision.transforms.ToTensor()),
                                               dm=opt.data_multiplication, use_all=True, label=True)

    paired_loader = torch.utils.data.DataLoader(paired_dataset,
                                                batch_size=opt.batch_size,
                                                num_workers=opt.n_cpu,
                                                shuffle=False,
                                                pin_memory=True,
                                                drop_last=False)

    conditional = models.DeterministicConditional if opt.deterministic else models.GaussianConditional

    mnist_encoder = conditional(
        models.mnist.Encoder(img_shape=mnist_img_shape,
                             latent_dim=opt.latent_dim if opt.deterministic else 2 * opt.latent_dim))
    mnist_decoder = models.mnist.Decoder(img_shape=mnist_img_shape,
                                         latent_dim=opt.latent_dim)

    svhn_encoder = conditional(
        models.svhn.Encoder(channels=svhn_channels,
                            latent_dim=opt.latent_dim if opt.deterministic else 2 * opt.latent_dim))
    svhn_decoder = models.svhn.Decoder(channels=svhn_channels,
                                       latent_dim=opt.latent_dim)

    mnist_classifier, svhn_classifier = load_or_train_classifier()

    mnist_encoder.to(device)
    mnist_decoder.to(device)
    svhn_encoder.to(device)
    svhn_decoder.to(device)

    mnist_encoder.load_state_dict((torch.load(os.path.join(opt.checkpoint_dir, 'mnist_encoder.pt'),
                                              map_location=device)))
    mnist_decoder.load_state_dict((torch.load(os.path.join(opt.checkpoint_dir, 'mnist_decoder.pt'),
                                              map_location=device)))
    svhn_encoder.load_state_dict((torch.load(os.path.join(opt.checkpoint_dir, 'svhn_encoder.pt'),
                                             map_location=device)))
    svhn_decoder.load_state_dict((torch.load(os.path.join(opt.checkpoint_dir, 'svhn_decoder.pt'),
                                             map_location=device)))

    mnist_encoder.eval()
    mnist_decoder.eval()
    mnist_classifier.eval()
    svhn_encoder.eval()
    svhn_decoder.eval()
    svhn_classifier.eval()

    with torch.no_grad():
        if not opt.deterministic:
            acc_syn_m, acc_syn_s = calc_synergy_coherence(paired_loader, mnist_encoder, mnist_decoder, mnist_classifier,
                                                          svhn_encoder, svhn_decoder, svhn_classifier)
            print('{:.6f}'.format(acc_syn_m))
            print('{:.6f}'.format(acc_syn_s))

        acc_m2s, acc_s2m = calc_cross_coherence(paired_loader, mnist_encoder, mnist_decoder, mnist_classifier,
                                                svhn_encoder, svhn_decoder, svhn_classifier)
        print('{:.6f}'.format(acc_m2s))
        print('{:.6f}'.format(acc_s2m))

        acc_joint = calc_joint_coherence(mnist_decoder, mnist_classifier, svhn_decoder, svhn_classifier)
        print('{:.6f}'.format(acc_joint))


if __name__ == '__main__':
    main()
