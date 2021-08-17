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
import models.cub_caption_image
import models.cub_image
import models.cub_caption
import options
import utils

opt = options.parser.parse_args()
print(opt)

key_cap = '0cap'
key_img = '1img'
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
def log_entropy(model, x1, x2, n_iter):
    model.eval()
    cap_encoder = getattr(model.encoders, key_cap).module
    img_encoder = getattr(model.encoders, key_img).module

    z1 = cap_encoder(x1)
    z2 = img_encoder(x2)
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
            key_cap: utils.calc_gaussian_entropy(style_logvar1).mean(dim=0),
            key_img: utils.calc_gaussian_entropy(style_logvar2).mean(dim=0),
        }, global_step=n_iter)

        writer.add_scalars('content_entropy', {
            key_cap: utils.calc_gaussian_entropy(content_logvar1).mean(dim=0),
            key_img: utils.calc_gaussian_entropy(content_logvar2).mean(dim=0),
            'avg': utils.calc_gaussian_entropy(content_logvar_avg).mean(dim=0),
            'poe': utils.calc_gaussian_entropy(content_logvar_poe).mean(dim=0),
        }, global_step=n_iter)
    else:
        writer.add_scalars('content_entropy', {
            key_cap: utils.calc_gaussian_entropy(logvar1).mean(dim=0),
            key_img: utils.calc_gaussian_entropy(logvar2).mean(dim=0),
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

    torch.save(getattr(model_ema.encoders, key_cap).state_dict(),
               os.path.join(output_dir, 'cap_encoder.pt'))

    torch.save(getattr(model_ema.decoders, key_cap).state_dict(),
               os.path.join(output_dir, 'cap_decoder.pt'))

    torch.save(getattr(model_ema.encoders, key_img).state_dict(),
               os.path.join(output_dir, 'img_encoder.pt'))

    torch.save(getattr(model_ema.decoders, key_img).state_dict(),
               os.path.join(output_dir, 'img_decoder.pt'))


def main():
    x1_dataset = datasets.CUBCaptionVector(opt.dataroot, model='fasttext', split='train', normalization='min-max')
    x2_dataset = datasets.CUBImageFeature(opt.dataroot, split='train', normalization='min-max')
    paired_dataset = datasets.CaptionImagePair(x1_dataset, x2_dataset)

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

    x1_discriminator = models.cub_caption.XZDiscriminator(latent_dim=opt.latent_dim, output_dim=4)
    x2_discriminator = models.cub_image.XZDiscriminatorFT(latent_dim=opt.latent_dim, output_dim=4)
    joint_discriminator = models.cub_caption_image.XXDiscriminatorFT()

    model = models.mmali.FactorModel(
        encoders={
            key_cap:
                conditional(
                    models.cub_caption.Encoder(latent_dim=factor * opt.latent_dim)),
            key_img:
                conditional(
                    models.cub_image.EncoderFT(latent_dim=factor * opt.latent_dim)),
        },
        decoders={
            key_cap:
                models.cub_caption.Decoder(latent_dim=opt.latent_dim),
            key_img:
                models.cub_image.DecoderFT(latent_dim=opt.latent_dim),
        },
        xz_discriminators={
            key_cap: x1_discriminator,
            key_img: x2_discriminator,
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

    decoded_fixed_x1 = x1_dataset.decode(fixed_x1)

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
                key_cap: {
                    'x': x1,
                    'z': torch.randn(opt.batch_size, opt.latent_dim).to(device),
                },
                key_img: {
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
                key_cap: {
                    'x': x1,
                    'z': torch.cat([s1, c], dim=1),
                    'extra_x': unpaired_x1,
                },
                key_img: {
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
        g_losses = {}

        x1, x2 = next(paired_dataloader)
        # x1, _ = next(x1_dataloader)
        # x2, _ = next(x2_dataloader)
        # x1, _ = next(x1_subsetloader)
        # x2, _ = next(x2_subsetloader)

        x1, x2 = x1.to(device), x2.to(device)
        g_losses.update(model({
            key_cap: {
                'x': x1,
                'z': torch.randn(opt.batch_size, opt.latent_dim).to(device),
            },
            key_img: {
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
            key_cap: {
                'x': x1,
                'z': torch.cat([s1, c], dim=1),
            },
            key_img: {
                'x': x2,
                'z': torch.cat([s2, c], dim=1),
            },
        }, train_d=False, joint=True, progress=progress))

        g_loss = sum(g_losses.values())
        optimizer_G.zero_grad(set_to_none=True)
        g_loss.backward()
        optimizer_G.step()

        d_loss = d_loss_iter_avg
        g_loss = g_loss.item()

        print(
            '[Iter {:d}/{:d}] [D loss: {:f}] [G loss: {:f}]'.format(n_iter, opt.max_iter, d_loss, g_loss),
        )

        d_losses = d_losses_iter_avg
        g_losses = {k: v.item() for k, v in g_losses.items()}
        writer.add_scalars('dis', d_losses, global_step=n_iter)
        writer.add_scalars('gen', g_losses, global_step=n_iter)
        writer.add_scalars('loss', {'d_loss': d_loss, 'g_loss': g_loss}, global_step=n_iter)

        if n_iter > opt.ema_start:
            model_ema.lerp(model, opt.beta)
        else:
            model_ema.lerp(model, 0.0)

        if n_iter % opt.save_interval == 0:
            with torch.no_grad():
                if not opt.deterministic:
                    paired_x1, paired_x2 = next(paired_dataloader)
                    paired_x1 = paired_x1.to(device)
                    paired_x2 = paired_x2.to(device)
                    log_entropy(model_ema, paired_x1, paired_x2, n_iter)

                cap_encoder = getattr(model_ema.encoders, key_cap)
                cap_decoder = getattr(model_ema.decoders, key_cap)

                img_encoder = getattr(model_ema.encoders, key_img)
                img_decoder = getattr(model_ema.decoders, key_img)

                enc_z1 = cap_encoder(fixed_x1)
                rec_x1 = cap_decoder(enc_z1)
                enc_z2 = img_encoder(fixed_x2)
                rec_x2 = img_decoder(enc_z2)
                cross_x1 = cap_decoder(torch.cat([enc_z1[:, :opt.style_dim], enc_z2[:, opt.style_dim:]], dim=1))
                cross_x2 = img_decoder(torch.cat([enc_z2[:, :opt.style_dim], enc_z1[:, opt.style_dim:]], dim=1))

                se1 = (rec_x1 - fixed_x1).reshape(rec_x1.size(0), -1).square()
                se2 = (rec_x2 - fixed_x2).reshape(rec_x2.size(0), -1).square()
                se3 = (cross_x1 - fixed_x1).reshape(rec_x1.size(0), -1).square()
                se4 = (cross_x2 - fixed_x2).reshape(rec_x2.size(0), -1).square()
                writer.add_scalars('sse', {
                    key_cap: se1.sum(dim=-1).mean(),
                    key_img: se2.sum(dim=-1).mean(),
                    'cross_{}'.format(key_cap): se3.sum(dim=-1).mean(),
                    'cross_{}'.format(key_img): se4.sum(dim=-1).mean(),
                }, global_step=n_iter)

                writer.add_scalars('mse', {
                    key_cap: se1.mean(),
                    key_img: se2.mean(),
                    'cross_{}'.format(key_cap): se3.mean(),
                    'cross_{}'.format(key_img): se4.mean(),
                }, global_step=n_iter)

                decoded_rec_x1 = x1_dataset.decode(rec_x1)
                decoded_cross_x1 = x1_dataset.decode(cross_x1)
                samples = ''
                filter_func = lambda w: w != '<pad>' and w != '<eos>'
                for fixed, rec, cross in zip(decoded_fixed_x1, decoded_rec_x1, decoded_cross_x1):
                    samples += ('input: ' + ' '.join(filter(filter_func, fixed)))
                    samples += '<br><br>'

                    samples += ('rec: ' + ' '.join(filter(filter_func, rec)))
                    samples += '<br><br>'

                    samples += ('cross: ' + ' '.join(filter(filter_func, cross)))
                    samples += '<br><br>'

                writer.add_text('samples', samples, global_step=n_iter)

                save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G)

                model.train()

    if not opt.deterministic:
        paired_x1, paired_x2 = next(paired_dataloader)
        paired_x1 = paired_x1.to(device)
        paired_x2 = paired_x2.to(device)
        log_entropy(model_ema, paired_x1, paired_x2, n_iter)

    save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G)


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
