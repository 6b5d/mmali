import copy
import os
import shutil
import subprocess

import torch.utils.data
import torch.utils.tensorboard
import torchvision.datasets

import datasets
import models.ali
import models.cub_image
import models.mnist
import models.svhn
import options
import utils

opt = options.parser.parse_args()
print(opt)

mnist_img_shape = (1, 28, 28)
svhn_channels = 3
cub_img_channels = 3
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
def save_samples(model, fixed_x, fixed_z, n_iter):
    model.eval()

    rec_x = model.decoder(model.encoder(fixed_x))
    gen_x = model.decoder(fixed_z)
    samples = torch.cat([fixed_x, rec_x, gen_x], dim=0)

    writer.add_image('samples',
                     torchvision.utils.make_grid(samples, nrow=n_samples, normalize=False),
                     global_step=n_iter)

    if opt.save_image:
        torchvision.utils.save_image(
            samples,
            '{}/samples_{:d}.png'.format(output_dir, n_iter), nrow=n_samples, normalize=False)


@torch.no_grad()
def log_entropy(model, x, n_iter):
    model.eval()

    enc_z = model.encoder.module(x)
    entropy = utils.calc_gaussian_entropy(enc_z[:, enc_z.size(1) // 2:]).mean(dim=0)
    writer.add_scalar('entropy', entropy.item(), global_step=n_iter)


def save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G):
    torch.save({
        'n_iter': n_iter,
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
    }, os.path.join(output_dir, 'checkpoint.pt'))
    torch.save(model_ema.encoder.state_dict(), os.path.join(output_dir, 'encoder.pt'))
    torch.save(model_ema.decoder.state_dict(), os.path.join(output_dir, 'decoder.pt'))


def eval_latent(n_iter):
    cmd = 'python {}/src/eval_mnist_svhn_latent.py'.format(output_dir)
    cmd += ' --n_epochs 100 --batch_size 256 --lr 1e-3 --b1 0.9 --b2 0.999'
    cmd += ' --dataroot {}'.format(opt.dataroot)
    cmd += ' --n_cpu {}'.format(opt.n_cpu)
    cmd += ' --latent_dim {}'.format(opt.latent_dim)
    cmd += ' --style_dim {}'.format(opt.style_dim)
    cmd += ' --dataset {}'.format(opt.dataset)
    cmd += ' --checkpoint {}/encoder.pt'.format(output_dir)

    if opt.deterministic:
        cmd += ' --deterministic'

    print('evaluating latent space:', cmd)
    accuracy = float(subprocess.run(cmd,
                                    capture_output=True,
                                    text=True,
                                    shell=True,
                                    cwd='{}/src'.format(output_dir),
                                    env=os.environ).stdout.strip().split('\n')[-1])
    print(accuracy)
    writer.add_scalar('latent', accuracy, global_step=n_iter)


def eval_prdc(n_iter):
    cmd = 'python {}/src/eval_mnist_svhn_prdc.py'.format(output_dir)
    cmd += ' --n_epochs 100 --batch_size 64'
    cmd += ' --dataroot {}'.format(opt.dataroot)
    cmd += ' --n_cpu {}'.format(opt.n_cpu)
    cmd += ' --latent_dim {}'.format(opt.latent_dim)
    cmd += ' --style_dim {}'.format(opt.style_dim)
    cmd += ' --dataset {}'.format(opt.dataset)
    cmd += ' --checkpoint {}/decoder.pt'.format(output_dir)

    print('evaluating prdc:', cmd)

    p, r, d, c = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                shell=True,
                                cwd='{}/src'.format(output_dir),
                                env=os.environ).stdout.strip().split('\n')[-4:]

    prdc = {'precision': float(p), 'recall': float(r), 'density': float(d), 'coverage': float(c)}

    writer.add_scalars('prdc', prdc, global_step=n_iter)


def main():
    if opt.dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(opt.dataroot,
                                             train=True,
                                             download=True,
                                             transform=torchvision.transforms.ToTensor())
        encoder = models.mnist.Encoder(img_shape=mnist_img_shape,
                                       latent_dim=opt.latent_dim if opt.deterministic else 2 * opt.latent_dim)
        decoder = models.mnist.Decoder(img_shape=mnist_img_shape,
                                       latent_dim=opt.latent_dim)
        discriminator = models.mnist.XZDiscriminator(img_shape=mnist_img_shape,
                                                     latent_dim=opt.latent_dim)
    elif opt.dataset == 'svhn':
        dataset = torchvision.datasets.SVHN(opt.dataroot,
                                            split='train',
                                            download=True,
                                            transform=torchvision.transforms.ToTensor())
        encoder = models.svhn.Encoder(channels=svhn_channels,
                                      latent_dim=opt.latent_dim if opt.deterministic else 2 * opt.latent_dim)
        decoder = models.svhn.Decoder(channels=svhn_channels,
                                      latent_dim=opt.latent_dim)
        discriminator = models.svhn.XZDiscriminator(channels=svhn_channels,
                                                    latent_dim=opt.latent_dim)
    else:
        raise NotImplementedError

    dataloader = iter(torch.utils.data.DataLoader(dataset,
                                                  batch_size=opt.batch_size,
                                                  num_workers=opt.n_cpu,
                                                  sampler=datasets.InfiniteSamplerWrapper(dataset),
                                                  pin_memory=True))

    conditional = models.DeterministicConditional if opt.deterministic else models.GaussianConditional
    encoder = conditional(encoder)

    model = models.ali.Model(encoder=encoder,
                             decoder=decoder,
                             discriminator=discriminator,
                             lambda_x_rec=opt.lambda_x_rec,
                             lambda_z_rec=opt.lambda_c_rec)

    utils.init_param_normal(model)
    model_ema = copy.deepcopy(model)

    model.to(device)
    model_ema.to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
    ], lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    fixed_x, _ = next(dataloader)
    fixed_x = fixed_x[:n_samples].to(device)
    fixed_z = torch.randn(n_samples, opt.latent_dim).to(device)

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
        progress = n_iter / (opt.max_iter - 1)
        # D update
        d_loss_iter_avg = 0.0
        d_losses_iter_avg = {}
        for _ in range(opt.dis_iter):
            d_losses = {}
            x, _ = next(dataloader)
            x = x.to(device)
            z = torch.randn(x.size(0), opt.latent_dim).to(device)

            d_losses.update(model(x, z, train_d=True, progress=progress))

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
            d_losses_iter_avg[k] /= opt.dis_iter

        # G update
        g_loss_iter_avg = 0.0
        g_losses_iter_avg = {}
        for _ in range(opt.gen_iter):
            g_losses = {}
            x, _ = next(dataloader)
            x = x.to(device)
            z = torch.randn(x.size(0), opt.latent_dim).to(device)
            g_losses.update(model(x, z, train_d=False, progress=progress))
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
            g_losses_iter_avg[k] /= opt.gen_iter

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
                log_entropy(model_ema, x, n_iter)

            save_samples(model_ema, fixed_x, fixed_z, n_iter)
            save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G)

            model.train()

        if not opt.no_eval and n_iter > 0 and n_iter % opt.eval_interval == 0:
            try:
                eval_latent(n_iter)
            except:
                print('Something wrong during evaluation')

    if not opt.deterministic:
        x, _ = next(dataloader)
        x = x.to(device)
        log_entropy(model_ema, x, n_iter)

    save_samples(model_ema, fixed_x, fixed_z, n_iter)
    save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G)

    if not opt.no_eval:
        try:
            eval_latent(n_iter)
        except:
            print('Something wrong during evaluation')


if __name__ == '__main__':
    main()
