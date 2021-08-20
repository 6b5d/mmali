import copy
import os
import shutil
import subprocess

import numpy as np
import torch
import torch.utils.data
import torch.utils.tensorboard
import torchvision.datasets
import torchvision.transforms

import datasets
import models
import models.mmali
import models.mnist
import models.mnist_svhn
import models.svhn
import options
import utils

opt = options.parser.parse_args()
print(opt)

mnist_img_shape = (1, 28, 28)
svhn_channels = 3
key_mnist = '0mnist'
key_svhn = '1svhn'
n_samples = min(32, opt.batch_size)

output_dir = os.path.join(opt.outroot,
                          os.path.basename(__file__).replace('train', 'exp').replace('.py', '') + '_' + opt.name)
os.makedirs(output_dir, exist_ok=True)
print('results will be saved to {}'.format(output_dir))
shutil.copytree(os.path.dirname(os.path.realpath(__file__)), os.path.join(output_dir, 'src'))
with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
    print(opt, file=f)
writer = torch.utils.tensorboard.SummaryWriter(log_dir=output_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def save_samples(model, fixed_x1, fixed_x2, fixed_s1, fixed_s2, fixed_c, n_iter):
    model.eval()
    mnist_encoder = getattr(model.encoders, key_mnist)
    svhn_encoder = getattr(model.encoders, key_svhn)
    mnist_decoder = getattr(model.decoders, key_mnist)
    svhn_decoder = getattr(model.decoders, key_svhn)

    x1_samples = [fixed_x1]
    x2_samples = [fixed_x2]

    if not opt.deterministic:
        mnist_encoder = mnist_encoder.module
        svhn_encoder = svhn_encoder.module
        z1 = mnist_encoder(fixed_x1)
        z2 = svhn_encoder(fixed_x2)
        avg_joint_z = utils.joint_posterior(z1, z2, average=True)
        poe_joint_z = utils.joint_posterior(z1, z2, average=False)

        z1_sample = utils.reparameterize(z1)
        z2_sample = utils.reparameterize(z2)
        avg_joint_z_sample = utils.reparameterize(avg_joint_z)
        poe_joint_z_sample = utils.reparameterize(poe_joint_z)
    else:
        z1_sample = mnist_encoder(fixed_x1)
        z2_sample = svhn_encoder(fixed_x2)

    if opt.style_dim > 0:
        enc_s1 = z1_sample[:, :opt.style_dim]
        enc_c1 = z1_sample[:, opt.style_dim:]

        enc_s2 = z2_sample[:, :opt.style_dim]
        enc_c2 = z2_sample[:, opt.style_dim:]

        if not opt.deterministic:
            avg_joint_c = avg_joint_z_sample[:, opt.style_dim:]
            poe_joint_c = poe_joint_z_sample[:, opt.style_dim:]

        # enc_style, x1
        enc_style_rec_x1 = mnist_decoder(torch.cat([enc_s1, enc_c1], dim=1))
        enc_style_cross_rec_x1 = mnist_decoder(torch.cat([enc_s1, enc_c2], dim=1))
        if not opt.deterministic:
            enc_style_avg_joint_rec_x1 = mnist_decoder(torch.cat([enc_s1, avg_joint_c], dim=1))
            enc_style_poe_joint_rec_x1 = mnist_decoder(torch.cat([enc_s1, poe_joint_c], dim=1))
        enc_style_joint_gen_x1 = mnist_decoder(torch.cat([enc_s1, fixed_c], dim=1))

        # fixed_style, x1
        fixed_style_rec_x1 = mnist_decoder(torch.cat([fixed_s1, enc_c1], dim=1))
        fixed_style_cross_rec_x1 = mnist_decoder(torch.cat([fixed_s1, enc_c2], dim=1))
        if not opt.deterministic:
            fixed_style_avg_joint_rec_x1 = mnist_decoder(torch.cat([fixed_s1, avg_joint_c], dim=1))
            fixed_style_poe_joint_rec_x1 = mnist_decoder(torch.cat([fixed_s1, poe_joint_c], dim=1))
        fixed_style_joint_gen_x1 = mnist_decoder(torch.cat([fixed_s1, fixed_c], dim=1))

        # enc_style, x2
        enc_style_rec_x2 = svhn_decoder(torch.cat([enc_s2, enc_c2], dim=1))
        enc_style_cross_rec_x2 = svhn_decoder(torch.cat([enc_s2, enc_c1], dim=1))
        if not opt.deterministic:
            enc_style_avg_joint_rec_x2 = svhn_decoder(torch.cat([enc_s2, avg_joint_c], dim=1))
            enc_style_poe_joint_rec_x2 = svhn_decoder(torch.cat([enc_s2, poe_joint_c], dim=1))
        enc_style_joint_gen_x2 = svhn_decoder(torch.cat([enc_s2, fixed_c], dim=1))

        # fixed_style, x2
        fixed_style_rec_x2 = svhn_decoder(torch.cat([fixed_s2, enc_c2], dim=1))
        fixed_style_cross_rec_x2 = svhn_decoder(torch.cat([fixed_s2, enc_c1], dim=1))
        if not opt.deterministic:
            fixed_style_avg_joint_rec_x2 = svhn_decoder(torch.cat([fixed_s2, avg_joint_c], dim=1))
            fixed_style_poe_joint_rec_x2 = svhn_decoder(torch.cat([fixed_s2, poe_joint_c], dim=1))
        fixed_joint_gen_x2 = svhn_decoder(torch.cat([fixed_s2, fixed_c], dim=1))

        if not opt.deterministic:
            x1_samples += [
                enc_style_rec_x1, enc_style_cross_rec_x1, enc_style_avg_joint_rec_x1,
                enc_style_poe_joint_rec_x1,

                fixed_style_rec_x1, fixed_style_cross_rec_x1, fixed_style_avg_joint_rec_x1,
                fixed_style_poe_joint_rec_x1,

                enc_style_joint_gen_x1, fixed_style_joint_gen_x1
            ]

            x2_samples += [
                enc_style_rec_x2, enc_style_cross_rec_x2, enc_style_avg_joint_rec_x2,
                enc_style_poe_joint_rec_x2,

                fixed_style_rec_x2, fixed_style_cross_rec_x2, fixed_style_avg_joint_rec_x2,
                fixed_style_poe_joint_rec_x2,

                enc_style_joint_gen_x2, fixed_joint_gen_x2
            ]
        else:
            x1_samples += [
                enc_style_rec_x1, enc_style_cross_rec_x1,
                fixed_style_rec_x1, fixed_style_cross_rec_x1,
                enc_style_joint_gen_x1, fixed_style_joint_gen_x1
            ]

            x2_samples += [
                enc_style_rec_x2, enc_style_cross_rec_x2,
                fixed_style_rec_x2, fixed_style_cross_rec_x2,
                enc_style_joint_gen_x2, fixed_joint_gen_x2
            ]
    else:
        enc_c1 = z1_sample
        enc_c2 = z2_sample
        avg_joint_c = avg_joint_z_sample
        poe_joint_c = poe_joint_z_sample

        # x1
        rec_x1 = mnist_decoder(enc_c1)
        cross_rec_x1 = mnist_decoder(enc_c2)
        avg_joint_rec_x1 = mnist_decoder(avg_joint_c)
        poe_joint_rec_x1 = mnist_decoder(poe_joint_c)
        joint_gen_x1 = mnist_decoder(fixed_c)

        # x2
        rec_x2 = svhn_decoder(enc_c2)
        cross_rec_x2 = svhn_decoder(enc_c1)
        avg_joint_rec_x2 = svhn_decoder(avg_joint_c)
        poe_joint_rec_x2 = svhn_decoder(poe_joint_c)
        joint_gen_x2 = svhn_decoder(fixed_c)

        x1_samples += [
            rec_x1, cross_rec_x1, avg_joint_rec_x1, poe_joint_rec_x1, joint_gen_x1
        ]

        x2_samples += [
            rec_x2, cross_rec_x2, avg_joint_rec_x2, poe_joint_rec_x2, joint_gen_x2
        ]

    x1_samples = torch.cat(x1_samples, dim=0)
    x2_samples = torch.cat(x2_samples, dim=0)

    writer.add_image('x1_samples',
                     torchvision.utils.make_grid(x1_samples, nrow=n_samples, normalize=False),
                     global_step=n_iter)
    writer.add_image('x2_samples',
                     torchvision.utils.make_grid(x2_samples, nrow=n_samples, normalize=False),
                     global_step=n_iter)

    if opt.save_image:
        torchvision.utils.save_image(
            x1_samples,
            '{}/x1_samples_{:d}.png'.format(output_dir, n_iter), nrow=n_samples, normalize=False)
        torchvision.utils.save_image(
            x2_samples,
            '{}/x2_samples_{:d}.png'.format(output_dir, n_iter), nrow=n_samples, normalize=False)


@torch.no_grad()
def log_entropy(model, x1, x2, n_iter):
    model.eval()
    mnist_encoder = getattr(model.encoders, key_mnist).module
    svhn_encoder = getattr(model.encoders, key_svhn).module

    z1 = mnist_encoder(x1)
    z2 = svhn_encoder(x2)
    avg_joint_z = utils.joint_posterior(z1, z2, average=True)
    poe_joint_z = utils.joint_posterior(z1, z2, average=False)
    ch = z1.size(1) // 2

    logvar1 = z1[:, ch:]
    logvar2 = z2[:, ch:]
    logvar_avg = avg_joint_z[:, ch:]
    logvar_poe = poe_joint_z[:, ch:]
    if opt.style_dim > 0:
        style_logvar1 = logvar1[:, :opt.style_dim]
        style_logvar2 = logvar2[:, :opt.style_dim]

        content_logvar1 = logvar1[:, opt.style_dim:]
        content_logvar2 = logvar2[:, opt.style_dim:]
        content_logvar_avg = logvar_avg[:, opt.style_dim:]
        content_logvar_poe = logvar_poe[:, opt.style_dim:]

        writer.add_scalars('style_entropy', {
            key_mnist: utils.calc_gaussian_entropy(style_logvar1).mean(dim=0),
            key_svhn: utils.calc_gaussian_entropy(style_logvar2).mean(dim=0),
        }, global_step=n_iter)

        writer.add_scalars('content_entropy', {
            key_mnist: utils.calc_gaussian_entropy(content_logvar1).mean(dim=0),
            key_svhn: utils.calc_gaussian_entropy(content_logvar2).mean(dim=0),
            'avg': utils.calc_gaussian_entropy(content_logvar_avg).mean(dim=0),
            'poe': utils.calc_gaussian_entropy(content_logvar_poe).mean(dim=0),
        }, global_step=n_iter)
    else:
        writer.add_scalars('content_entropy', {
            key_mnist: utils.calc_gaussian_entropy(logvar1).mean(dim=0),
            key_svhn: utils.calc_gaussian_entropy(logvar2).mean(dim=0),
            'avg': utils.calc_gaussian_entropy(logvar_avg).mean(dim=0),
            'poe': utils.calc_gaussian_entropy(logvar_poe).mean(dim=0),
        }, global_step=n_iter)


def save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G):
    torch.save({
        'n_iter': n_iter,
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
    }, os.path.join(output_dir, 'checkpoint.pt'))

    torch.save(getattr(model_ema.encoders, key_mnist).state_dict(),
               os.path.join(output_dir, 'mnist_encoder.pt'))

    torch.save(getattr(model_ema.decoders, key_mnist).state_dict(),
               os.path.join(output_dir, 'mnist_decoder.pt'))

    torch.save(getattr(model_ema.encoders, key_svhn).state_dict(),
               os.path.join(output_dir, 'svhn_encoder.pt'))

    torch.save(getattr(model_ema.decoders, key_svhn).state_dict(),
               os.path.join(output_dir, 'svhn_decoder.pt'))


def eval_latent(n_iter):
    cmd = 'python {}/src/eval_mnist_svhn_latent.py'.format(output_dir)
    cmd += ' --n_epochs 100 --batch_size 256 --lr 1e-3 --b1 0.9 --b2 0.999'
    cmd += ' --dataroot {}'.format(opt.dataroot)
    cmd += ' --n_cpu {}'.format(opt.n_cpu)
    cmd += ' --latent_dim {}'.format(opt.latent_dim)
    cmd += ' --style_dim {}'.format(opt.style_dim)

    if opt.deterministic:
        cmd += ' --deterministic'

    accuracies = {}
    for dset in ['mnist', 'svhn']:
        extra = ' --dataset {}' \
                ' --checkpoint {}/{}_encoder.pt'.format(dset, output_dir, dset)
        print('evaluating latent space:', cmd + extra)
        accuracies[dset] = float(subprocess.run(cmd + extra,
                                                capture_output=True,
                                                text=True,
                                                shell=True,
                                                cwd='{}/src'.format(output_dir),
                                                env=os.environ).stdout.strip().split('\n')[-1])
    print(accuracies)
    writer.add_scalars('latent', accuracies, global_step=n_iter)


def eval_generation(n_iter):
    cmd = 'python {}/src/eval_mnist_svhn_gen.py'.format(output_dir)
    cmd += ' --n_epochs 100 --batch_size 256 --lr 1e-3 --b1 0.9 --b2 0.999'
    cmd += ' --dataroot {}'.format(opt.dataroot)
    cmd += ' --n_cpu {}'.format(opt.n_cpu)
    cmd += ' --style_dim {}'.format(opt.style_dim)
    cmd += ' --latent_dim {}'.format(opt.latent_dim)
    cmd += ' --checkpoint_dir {}'.format(output_dir)
    if opt.deterministic:
        cmd += ' --deterministic'

    print('evaluating generation:', cmd)

    if opt.deterministic:
        acc_m2s, acc_s2m, acc_joint = subprocess.run(cmd,
                                                     capture_output=True, text=True, shell=True,
                                                     cwd='{}/src'.format(output_dir),
                                                     env=os.environ).stdout.strip().split('\n')[-3:]

        acc_m2s = float(acc_m2s)
        acc_s2m = float(acc_s2m)
        acc_joint = float(acc_joint)
        accuracies = {
            'm2s': acc_m2s,
            's2m': acc_s2m,
            'joint': acc_joint,
        }
    else:
        acc_syn_m, acc_syn_s, acc_m2s, acc_s2m, acc_joint = subprocess.run(cmd,
                                                                           capture_output=True, text=True, shell=True,
                                                                           cwd='{}/src'.format(output_dir),
                                                                           env=os.environ).stdout.strip().split('\n')[
                                                            -5:]

        acc_syn_m = float(acc_syn_m)
        acc_syn_s = float(acc_syn_s)
        acc_m2s = float(acc_m2s)
        acc_s2m = float(acc_s2m)
        acc_joint = float(acc_joint)
        accuracies = {
            'syn_m': acc_syn_m,
            'syn_s': acc_syn_s,
            'm2s': acc_m2s,
            's2m': acc_s2m,
            'joint': acc_joint,
        }

    print(accuracies)

    writer.add_scalars('generation', accuracies, global_step=n_iter)


def eval_prdc(n_iter):
    cmd = 'python {}/src/eval_mnist_svhn_prdc.py'.format(output_dir)
    cmd += ' --n_epochs 100 --batch_size 64'
    cmd += ' --dataroot {}'.format(opt.dataroot)
    cmd += ' --n_cpu {}'.format(opt.n_cpu)
    cmd += ' --latent_dim {}'.format(opt.latent_dim)
    cmd += ' --style_dim {}'.format(opt.style_dim)

    prdcs = {}
    for dset in ['mnist', 'svhn']:
        extra = ' --dataset {}' \
                ' --checkpoint {}/{}_decoder.pt'.format(dset, output_dir, dset)
        print('evaluating prdc:', cmd + extra)
        p, r, d, c = subprocess.run(cmd + extra,
                                    capture_output=True,
                                    text=True,
                                    shell=True,
                                    cwd='{}/src'.format(output_dir),
                                    env=os.environ).stdout.strip().split('\n')[-4:]

        prdcs[dset] = {'precision': float(p), 'recall': float(r), 'density': float(d), 'coverage': float(c)}
    print(prdcs)
    writer.add_scalars('prdc_mnist', prdcs['mnist'], global_step=n_iter)
    writer.add_scalars('prdc_svhn', prdcs['svhn'], global_step=n_iter)


def main():
    x1_dataset = torchvision.datasets.MNIST(opt.dataroot, train=True, download=True,
                                            transform=torchvision.transforms.ToTensor())
    x2_dataset = torchvision.datasets.SVHN(opt.dataroot, split='train', download=True,
                                           transform=torchvision.transforms.ToTensor())
    paired_dataset = datasets.PairedMNISTSVHN2(x1_dataset, x2_dataset,
                                               max_d=opt.max_d, dm=opt.data_multiplication, use_all=False)

    paired_idx1 = paired_dataset.mnist_idx.unique().numpy()
    diff = np.setdiff1d(np.arange(len(x1_dataset)), paired_idx1)
    print('x1 remain: {} - {} = {}'.format(len(diff), opt.n_extra_x1, len(diff) - opt.n_extra_x1))
    if opt.n_extra_x1 > 0:
        extra_idx1 = diff[:opt.n_extra_x1]
        total_idx1 = np.concatenate([paired_idx1, extra_idx1])
    else:
        total_idx1 = paired_idx1

    # x1_subset = torch.utils.data.Subset(x1_dataset, extra_idx1)
    x1_universal_set = torch.utils.data.Subset(x1_dataset, total_idx1)

    paired_idx2 = paired_dataset.svhn_idx.unique()
    diff = np.setdiff1d(np.arange(len(x2_dataset)), paired_idx2)
    print('x2 remain: {} - {} = {}'.format(len(diff), opt.n_extra_x2, len(diff) - opt.n_extra_x2))
    if opt.n_extra_x2 > 0:
        extra_idx2 = diff[:opt.n_extra_x2]
        total_idx2 = np.concatenate([paired_idx2, extra_idx2])
    else:
        total_idx2 = paired_idx2

    # x2_subset = torch.utils.data.Subset(x2_dataset, extra_idx2)
    x2_universal_set = torch.utils.data.Subset(x2_dataset, total_idx2)

    x1_dataloader = iter(
        torch.utils.data.DataLoader(x1_universal_set,
                                    batch_size=opt.batch_size,
                                    num_workers=opt.n_cpu,
                                    sampler=datasets.InfiniteSamplerWrapper(x1_universal_set),
                                    pin_memory=True))
    # x1_subsetloader = iter(
    #     torch.utils.data.DataLoader(x1_subset,
    #                                 batch_size=opt.batch_size,
    #                                 num_workers=opt.n_cpu,
    #                                 sampler=datasets.InfiniteSamplerWrapper(x1_subset),
    #                                 pin_memory=True))
    x2_dataloader = iter(
        torch.utils.data.DataLoader(x2_universal_set,
                                    batch_size=opt.batch_size,
                                    num_workers=opt.n_cpu,
                                    sampler=datasets.InfiniteSamplerWrapper(x2_universal_set),
                                    pin_memory=True))
    # x2_subsetloader = iter(
    #     torch.utils.data.DataLoader(x2_subset,
    #                                 batch_size=opt.batch_size,
    #                                 num_workers=opt.n_cpu,
    #                                 sampler=datasets.InfiniteSamplerWrapper(x2_subset),
    #                                 pin_memory=True))
    paired_dataloader = iter(
        torch.utils.data.DataLoader(paired_dataset,
                                    batch_size=opt.batch_size,
                                    num_workers=opt.n_cpu,
                                    sampler=datasets.InfiniteSamplerWrapper(paired_dataset),
                                    pin_memory=True))

    if opt.deterministic:
        conditional = models.DeterministicConditional
        factor = 1
    else:
        conditional = models.GaussianConditional
        factor = 2
    content_dim = opt.latent_dim - opt.style_dim

    # x1_feature = models.mnist.XDiscriminationFeature(img_shape=mnist_img_shape)
    # x2_feature = models.svhn.XDiscriminationFeature(channels=svhn_channels)
    # x1_discriminator = models.mnist.XFeatureZDiscriminator(x1_feature, opt.latent_dim, output_dim=4)
    # x2_discriminator = models.svhn.XFeatureZDiscriminator(x2_feature, opt.latent_dim, output_dim=4)
    # joint_discriminator = models.mnist_svhn.XXFeatureDiscriminator(x1_feature, x2_feature)

    x1_discriminator = models.mnist.XZDiscriminator(latent_dim=opt.latent_dim, img_shape=mnist_img_shape, output_dim=4)
    x2_discriminator = models.svhn.XZDiscriminator(latent_dim=opt.latent_dim, channels=svhn_channels, output_dim=4)
    joint_discriminator = models.mnist_svhn.XXDiscriminator(img_shape=mnist_img_shape, channels=svhn_channels)
    # joint_discriminator = models.mnist_svhn.XXDiscriminatorConv()

    model = models.mmali.FactorModel(
        encoders={
            key_mnist:
                conditional(
                    models.mnist.Encoder(img_shape=mnist_img_shape, latent_dim=factor * opt.latent_dim)
                ),
            key_svhn:
                conditional(
                    models.svhn.Encoder(channels=svhn_channels, latent_dim=factor * opt.latent_dim)
                ),
        },
        decoders={
            key_mnist:
                models.mnist.Decoder(img_shape=mnist_img_shape, latent_dim=opt.latent_dim),
            key_svhn:
                models.svhn.Decoder(channels=svhn_channels, latent_dim=opt.latent_dim),
        },
        xz_discriminators={
            key_mnist: x1_discriminator,
            key_svhn: x2_discriminator,
        },
        joint_discriminator=joint_discriminator,
        content_dim=content_dim,
        lambda_unimodal=opt.lambda_unimodal,
        lambda_x_rec=opt.lambda_x_rec,
        lambda_c_rec=opt.lambda_c_rec,
        lambda_s_rec=opt.lambda_s_rec,
    )

    utils.init_param_normal(model)
    model_ema = copy.deepcopy(model)

    model.to(device)
    model_ema.to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(model.generators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(model.discriminators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    fixed_x1, fixed_x2 = next(paired_dataloader)
    fixed_x1 = fixed_x1[:n_samples].to(device)
    fixed_x2 = fixed_x2[:n_samples].to(device)
    fixed_s1 = torch.randn(n_samples, opt.style_dim).to(device)
    fixed_s2 = torch.randn(n_samples, opt.style_dim).to(device)
    fixed_c = torch.randn(n_samples, content_dim).to(device)

    start_iter = 0
    if opt.checkpoint is not None:
        checkpoint = torch.load(opt.checkpoint, map_location=device)
        start_iter = checkpoint['n_iter']
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        print('resumed from {}'.format(opt.checkpoint))

    model.train()
    for n_iter in range(start_iter, opt.max_iter):
        # progress range: [0, 1]
        progress = n_iter / (opt.max_iter - 1)
        # D update
        d_loss_iter_avg = 0.0
        d_losses_iter_avg = {}
        for _ in range(opt.dis_iter):
            d_losses = {}

            x1, x2 = next(paired_dataloader)
            # x1, _ = next(x1_dataloader)
            # x2, _ = next(x2_dataloader)

            x1, x2 = x1.to(device), x2.to(device)

            d_losses.update(model({
                key_mnist: {
                    'x': x1,
                    'z': torch.randn(opt.batch_size, opt.latent_dim).to(device),
                },
                key_svhn: {
                    'x': x2,
                    'z': torch.randn(opt.batch_size, opt.latent_dim).to(device)
                },
            }, train_d=True, joint=False, progress=progress))

            x1, x2 = next(paired_dataloader)
            unpaired_x1 = utils.permute_dim(x1, dim=0)
            unpaired_x2 = utils.permute_dim(x2, dim=0)

            # unpaired_x1, _ = next(x1_dataloader)
            # unpaired_x2, _ = next(x2_dataloader)

            x1, x2 = x1.to(device), x2.to(device)
            unpaired_x1, unpaired_x2 = unpaired_x1.to(device), unpaired_x2.to(device)

            s1 = torch.randn(opt.batch_size, opt.style_dim).to(device)
            s2 = torch.randn(opt.batch_size, opt.style_dim).to(device)
            c = torch.randn(opt.batch_size, opt.latent_dim - opt.style_dim).to(device)

            d_losses.update(model({
                key_mnist: {
                    'x': x1,
                    'z': torch.cat([s1, c], dim=1),
                    'extra_x': unpaired_x1,
                },
                key_svhn: {
                    'x': x2,
                    'z': torch.cat([s2, c], dim=1),
                    'extra_x': unpaired_x2,
                },
            }, train_d=True, joint=True, progress=progress))

            d_loss = sum(d_losses.values())
            optimizer_D.zero_grad(set_to_none=True)
            d_loss.backward()
            optimizer_D.step()

            d_loss_iter_avg += d_loss.item()
            for k in d_losses.keys():
                if k in d_losses_iter_avg:
                    d_losses_iter_avg[k] += d_losses[k].item()
                else:
                    d_losses_iter_avg[k] = d_losses[k].item()

        d_loss_iter_avg /= opt.dis_iter
        for k in d_losses_iter_avg.keys():
            d_losses_iter_avg[k] = d_losses_iter_avg[k] / opt.dis_iter

        # G update
        g_loss_iter_avg = 0.0
        g_losses_iter_avg = {}
        for _ in range(opt.gen_iter):
            g_losses = {}

            x1, x2 = next(paired_dataloader)
            # x1, _ = next(x1_dataloader)
            # x2, _ = next(x2_dataloader)
            # x1, _ = next(x1_subsetloader)
            # x2, _ = next(x2_subsetloader)

            x1, x2 = x1.to(device), x2.to(device)
            g_losses.update(model({
                key_mnist: {
                    'x': x1,
                    'z': torch.randn(opt.batch_size, opt.latent_dim).to(device),
                },
                key_svhn: {
                    'x': x2,
                    'z': torch.randn(opt.batch_size, opt.latent_dim).to(device)
                },
            }, train_d=False, joint=False, progress=progress))

            x1, x2 = next(paired_dataloader)
            x1, x2 = x1.to(device), x2.to(device)
            s1 = torch.randn(opt.batch_size, opt.style_dim).to(device)
            s2 = torch.randn(opt.batch_size, opt.style_dim).to(device)
            c = torch.randn(opt.batch_size, opt.latent_dim - opt.style_dim).to(device)
            g_losses.update(model({
                key_mnist: {
                    'x': x1,
                    'z': torch.cat([s1, c], dim=1),
                },
                key_svhn: {
                    'x': x2,
                    'z': torch.cat([s2, c], dim=1),
                },
            }, train_d=False, joint=True, progress=progress))

            g_loss = sum(g_losses.values())
            optimizer_G.zero_grad(set_to_none=True)
            g_loss.backward()
            optimizer_G.step()

            g_loss_iter_avg += g_loss.item()
            for k in g_losses.keys():
                if k in g_losses_iter_avg:
                    g_losses_iter_avg[k] += g_losses[k].item()
                else:
                    g_losses_iter_avg[k] = g_losses[k].item()

        g_loss_iter_avg /= opt.gen_iter
        for k in g_losses_iter_avg.keys():
            g_losses_iter_avg[k] = g_losses_iter_avg[k] / opt.gen_iter

        d_loss = d_loss_iter_avg
        g_loss = g_loss_iter_avg

        print(
            '[Iter {:d}/{:d}] [D loss: {:f}] [G loss: {:f}]'.format(n_iter, opt.max_iter, d_loss, g_loss),
        )

        d_losses = d_losses_iter_avg
        g_losses = g_losses_iter_avg
        writer.add_scalars('dis', d_losses, global_step=n_iter)
        writer.add_scalars('gen', g_losses, global_step=n_iter)
        writer.add_scalars('loss', {'d_loss': d_loss, 'g_loss': g_loss}, global_step=n_iter)

        if n_iter > opt.ema_start:
            model_ema.lerp(model, opt.beta)
        else:
            model_ema.lerp(model, 0.0)

        if n_iter % opt.save_interval == 0:
            if not opt.deterministic:
                paired_x1, paired_x2 = next(paired_dataloader)
                paired_x1 = paired_x1.to(device)
                paired_x2 = paired_x2.to(device)
                log_entropy(model_ema, paired_x1, paired_x2, n_iter)

            save_samples(model_ema, fixed_x1, fixed_x2, fixed_s1, fixed_s2, fixed_c, n_iter)
            save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G)

            model.train()

        if not opt.no_eval and n_iter > 0 and n_iter % opt.eval_interval == 0:
            try:
                eval_latent(n_iter)
                eval_generation(n_iter)
            except:
                print('Something wrong during evaluation')

    if not opt.deterministic:
        paired_x1, paired_x2 = next(paired_dataloader)
        paired_x1 = paired_x1.to(device)
        paired_x2 = paired_x2.to(device)
        log_entropy(model_ema, paired_x1, paired_x2, n_iter)

    save_samples(model_ema, fixed_x1, fixed_x2, fixed_s1, fixed_s2, fixed_c, n_iter)
    save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G)

    try:
        eval_latent(n_iter)
        eval_generation(n_iter)
    except:
        print('Something wrong during evaluation')


# default option
# lambda_x_rec 0.05
# lambda_c_rec 0.05
# lambda_s_rec 0.05
# lambda_unimodal 0.1
if __name__ == '__main__':
    # from mem import pre_occupy
    #
    # pre_occupy(percent=0.1)
    main()
