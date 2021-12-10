import copy
import os
import shutil

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import torchvision.datasets
import torchvision.transforms

import datasets
import models
import models.mmali
import models.mnist
import models.multimnist
import models.svhn
import options
import utils

opt = options.parser.parse_args()
print(opt)

mnist_img_shape = (1, 28, 28)
key_template = '{}mnist'
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
def save_samples(model, fixed_X, fixed_S, fixed_c, n_iter):
    model.eval()

    X_samples = [[x] for x in fixed_X]

    Z = [getattr(model.encoders, key_template.format(i)).module(fixed_X[i]) for i in range(opt.n_modalities)]
    avg_joint_z = utils.joint_posterior(*Z, average=True)
    poe_joint_z = utils.joint_posterior(*Z, average=False)

    Z_sample = [utils.reparameterize(z) for z in Z]
    avg_joint_z_sample = utils.reparameterize(avg_joint_z)
    poe_joint_z_sample = utils.reparameterize(poe_joint_z)

    if opt.style_dim > 0:

        enc_S = [z_sample[:, :opt.style_dim] for z_sample in Z_sample]
        enc_C = [z_sample[:, opt.style_dim:] for z_sample in Z_sample]

        avg_joint_c = avg_joint_z_sample[:, opt.style_dim:]
        poe_joint_c = poe_joint_z_sample[:, opt.style_dim:]

        for i in range(opt.n_modalities):
            decoder = getattr(model.decoders, key_template.format(i))
            enc_style_rec_x = decoder(torch.cat([enc_S[i], enc_C[i]], dim=1))
            enc_style_avg_joint_rec_x = decoder(torch.cat([enc_S[i], avg_joint_c], dim=1))
            enc_style_poe_joint_rec_x = decoder(torch.cat([enc_S[i], poe_joint_c], dim=1))
            enc_style_joint_gen_x = decoder(torch.cat([enc_S[i], fixed_c], dim=1))

            fixed_style_rec_x = decoder(torch.cat([fixed_S[i], enc_C[i]], dim=1))
            fixed_style_avg_joint_rec_x = decoder(torch.cat([fixed_S[i], avg_joint_c], dim=1))
            fixed_style_poe_joint_rec_x = decoder(torch.cat([fixed_S[i], poe_joint_c], dim=1))
            fixed_style_joint_gen_x = decoder(torch.cat([fixed_S[i], fixed_c], dim=1))

            X_samples[i].extend([
                enc_style_rec_x, enc_style_avg_joint_rec_x,
                enc_style_poe_joint_rec_x,

                fixed_style_rec_x, fixed_style_avg_joint_rec_x,
                fixed_style_poe_joint_rec_x,

                enc_style_joint_gen_x, fixed_style_joint_gen_x
            ])
    else:
        enc_C = Z_sample

        avg_joint_c = avg_joint_z_sample
        poe_joint_c = poe_joint_z_sample

        for i in range(opt.n_modalities):
            decoder = getattr(model.decoders, key_template.format(i))
            rec_x = decoder(enc_C[i])
            avg_joint_rec_x = decoder(avg_joint_c)
            poe_joint_rec_x = decoder(poe_joint_c)
            joint_gen_x = decoder(fixed_c)

            X_samples[i].extend([
                rec_x, avg_joint_rec_x,
                poe_joint_rec_x,
                joint_gen_x,
            ])

    X_samples = [torch.cat(x_samples) for x_samples in X_samples]

    for i in range(opt.n_modalities):
        writer.add_image('x{}_samples'.format(i),
                         torchvision.utils.make_grid(X_samples[i], nrow=n_samples, normalize=False),
                         global_step=n_iter)

    if opt.save_image:
        for i in range(opt.n_modalities):
            torchvision.utils.save_image(
                X_samples[i],
                '{}/x{}_samples_{:d}.png'.format(output_dir, i, n_iter), nrow=n_samples, normalize=False)
    model.train()


def save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G):
    torch.save({
        'n_iter': n_iter,
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
    }, os.path.join(output_dir, 'checkpoint.pt'))

    for i in range(opt.n_modalities):
        torch.save(getattr(model_ema.encoders, key_template.format(i)).state_dict(),
                   os.path.join(output_dir, 'mnist{}_encoder.pt'.format(i)))

        torch.save(getattr(model_ema.decoders, key_template.format(i)).state_dict(),
                   os.path.join(output_dir, 'mnist{}_decoder.pt'.format(i)))


def main():
    x_datasets = [torchvision.datasets.MNIST(opt.dataroot, train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())]

    for i in range(1, opt.n_modalities):
        x_datasets.append(torchvision.datasets.MNIST(opt.dataroot, train=True, download=True,
                                                     transform=torchvision.transforms.Compose([
                                                         torchvision.transforms.RandomRotation((i * 90, i * 90)),
                                                         torchvision.transforms.ToTensor()
                                                     ])))

    unpaired_dataloader = [iter(
        torch.utils.data.DataLoader(x_dataset,
                                    batch_size=opt.batch_size,
                                    num_workers=opt.n_cpu,
                                    sampler=datasets.InfiniteSamplerWrapper(x_dataset),
                                    pin_memory=True)) for x_dataset in x_datasets]

    paired_dataset = datasets.MultiMNIST(x_datasets, max_d=opt.max_d, dm=opt.data_multiplication)

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

    model = models.mmali.FactorModel(
        encoders={
            key_template.format(i):
                conditional(
                    models.mnist.Encoder(img_shape=mnist_img_shape, latent_dim=factor * opt.latent_dim))
            for i in range(opt.n_modalities)
        },

        decoders={
            key_template.format(i):
                models.mnist.Decoder(img_shape=mnist_img_shape, latent_dim=opt.latent_dim)
            for i in range(opt.n_modalities)
        },
        xz_discriminators={
            key_template.format(i):
                nn.ModuleList([
                    models.mnist.XZDiscriminator(img_shape=mnist_img_shape, latent_dim=opt.latent_dim,
                                                 output_dim=opt.n_modalities + 1),
                    models.mnist.XZDiscriminator(img_shape=mnist_img_shape, latent_dim=opt.latent_dim),
                ])
            for i in range(opt.n_modalities)
        },
        joint_discriminator=models.multimnist.MultiXDiscriminator(n_modalities=opt.n_modalities),
        content_dim=content_dim,
        lambda_unimodal=opt.lambda_unimodal,
        lambda_x_rec=opt.lambda_x_rec,
        lambda_c_rec=opt.lambda_c_rec,
        lambda_s_rec=opt.lambda_s_rec,
        joint_rec=opt.joint_rec,
    )

    utils.init_param_normal(model)
    model_ema = copy.deepcopy(model)

    model.to(device)
    model_ema.to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(model.generators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(model.discriminators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    fixed_X = next(paired_dataloader)
    fixed_X = [x[:n_samples].to(device) for x in fixed_X]
    fixed_S = [torch.randn(n_samples, opt.style_dim).to(device) for _ in range(opt.n_modalities)]
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

            X = next(paired_dataloader)
            # X = [next(loader)[0] for loader in unpaired_dataloader]

            X = [x.to(device) for x in X]

            d_losses.update(model({
                key_template.format(i): {
                    'x': X[i],
                    'z': torch.randn(opt.batch_size, opt.latent_dim).to(device)
                } for i in range(opt.n_modalities)
            }, train_d=True, joint=False, progress=progress))

            X = next(paired_dataloader)
            unpaired_X = [utils.permute_dim(x, dim=0) for x in X]
            # unpaired_X = [next(loader)[0] for loader in unpaired_dataloader]

            X = [x.to(device) for x in X]
            unpaired_X = [x.to(device) for x in unpaired_X]

            S = [torch.randn(opt.batch_size, opt.style_dim).to(device) for _ in range(opt.n_modalities)]
            c = torch.randn(opt.batch_size, opt.latent_dim - opt.style_dim).to(device)

            d_losses.update(model({
                key_template.format(i): {
                    'x': X[i],
                    'z': torch.cat([S[i], c], dim=1),
                    'extra_x': unpaired_X[i]
                } for i in range(opt.n_modalities)
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

            X = next(paired_dataloader)

            X = [x.to(device) for x in X]
            g_losses.update(model({
                key_template.format(i): {
                    'x': X[i],
                    'z': torch.randn(opt.batch_size, opt.latent_dim).to(device)
                } for i in range(opt.n_modalities)
            }, train_d=False, joint=False, progress=progress))

            X = next(paired_dataloader)

            X = [x.to(device) for x in X]

            S = [torch.randn(opt.batch_size, opt.style_dim).to(device) for _ in range(opt.n_modalities)]
            c = torch.randn(opt.batch_size, opt.latent_dim - opt.style_dim).to(device)
            g_losses.update(model({
                key_template.format(i): {
                    'x': X[i],
                    'z': torch.cat([S[i], c], dim=1),
                } for i in range(opt.n_modalities)
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
            model.eval()

            save_samples(model_ema, fixed_X, fixed_S, fixed_c, n_iter)
            save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G)

            model.train()

    save_samples(model_ema, fixed_X, fixed_S, fixed_c, n_iter)
    save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G)


if __name__ == '__main__':
    main()
