import copy
import os
import shutil
import subprocess

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard

import datasets
import models
import models.cub_caption
import models.cub_caption_image
import models.cub_image
import models.mmali
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
def save_samples(model, fixed_x1, fixed_x2, fixed_s1, fixed_s2, fixed_c, n_iter, cap_dataset):
    model.eval()
    cap_encoder = getattr(model.encoders, key_cap)
    img_encoder = getattr(model.encoders, key_img)
    cap_decoder = getattr(model.decoders, key_cap)
    img_decoder = getattr(model.decoders, key_img)

    x1_samples = [fixed_x1]
    x2_samples = [fixed_x2]

    if opt.deterministic:
        pass
    else:
        cap_encoder = cap_encoder.module
        img_encoder = img_encoder.module
        z1 = cap_encoder(fixed_x1)
        z2 = img_encoder(fixed_x2)
        avg_joint_z = utils.joint_posterior(z1, z2, average=True)
        poe_joint_z = utils.joint_posterior(z1, z2, average=False)

        z1_sample = utils.reparameterize(z1)
        z2_sample = utils.reparameterize(z2)
        avg_joint_z_sample = utils.reparameterize(avg_joint_z)
        poe_joint_z_sample = utils.reparameterize(poe_joint_z)

    if opt.style_dim > 0:
        enc_s1 = z1_sample[:, :opt.style_dim]
        enc_c1 = z1_sample[:, opt.style_dim:]

        enc_s2 = z2_sample[:, :opt.style_dim]
        enc_c2 = z2_sample[:, opt.style_dim:]

        if not opt.deterministic:
            avg_joint_c = avg_joint_z_sample[:, opt.style_dim:]
            poe_joint_c = poe_joint_z_sample[:, opt.style_dim:]

        # enc_style, x1
        enc_style_rec_x1 = cap_decoder(torch.cat([enc_s1, enc_c1], dim=1))
        enc_style_cross_rec_x1 = cap_decoder(torch.cat([enc_s1, enc_c2], dim=1))
        if not opt.deterministic:
            enc_style_avg_joint_rec_x1 = cap_decoder(torch.cat([enc_s1, avg_joint_c], dim=1))
            enc_style_poe_joint_rec_x1 = cap_decoder(torch.cat([enc_s1, poe_joint_c], dim=1))
        enc_style_joint_gen_x1 = cap_decoder(torch.cat([enc_s1, fixed_c], dim=1))

        # fixed_style, x1
        fixed_style_rec_x1 = cap_decoder(torch.cat([fixed_s1, enc_c1], dim=1))
        fixed_style_cross_rec_x1 = cap_decoder(torch.cat([fixed_s1, enc_c2], dim=1))
        if not opt.deterministic:
            fixed_style_avg_joint_rec_x1 = cap_decoder(torch.cat([fixed_s1, avg_joint_c], dim=1))
            fixed_style_poe_joint_rec_x1 = cap_decoder(torch.cat([fixed_s1, poe_joint_c], dim=1))
        fixed_style_joint_gen_x1 = cap_decoder(torch.cat([fixed_s1, fixed_c], dim=1))

        # enc_style, x2
        enc_style_rec_x2 = img_decoder(torch.cat([enc_s2, enc_c2], dim=1))
        enc_style_cross_rec_x2 = img_decoder(torch.cat([enc_s2, enc_c1], dim=1))
        if not opt.deterministic:
            enc_style_avg_joint_rec_x2 = img_decoder(torch.cat([enc_s2, avg_joint_c], dim=1))
            enc_style_poe_joint_rec_x2 = img_decoder(torch.cat([enc_s2, poe_joint_c], dim=1))
        enc_style_joint_gen_x2 = img_decoder(torch.cat([enc_s2, fixed_c], dim=1))

        # fixed_style, x2
        fixed_style_rec_x2 = img_decoder(torch.cat([fixed_s2, enc_c2], dim=1))
        fixed_style_cross_rec_x2 = img_decoder(torch.cat([fixed_s2, enc_c1], dim=1))
        if not opt.deterministic:
            fixed_style_avg_joint_rec_x2 = img_decoder(torch.cat([fixed_s2, avg_joint_c], dim=1))
            fixed_style_poe_joint_rec_x2 = img_decoder(torch.cat([fixed_s2, poe_joint_c], dim=1))
        fixed_joint_gen_x2 = img_decoder(torch.cat([fixed_s2, fixed_c], dim=1))

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
        rec_x1 = cap_decoder(enc_c1)
        cross_rec_x1 = cap_decoder(enc_c2)
        avg_joint_rec_x1 = cap_decoder(avg_joint_c)
        poe_joint_rec_x1 = cap_decoder(poe_joint_c)
        joint_gen_x1 = cap_decoder(fixed_c)

        # x2
        rec_x2 = img_decoder(enc_c2)
        cross_rec_x2 = img_decoder(enc_c1)
        avg_joint_rec_x2 = img_decoder(avg_joint_c)
        poe_joint_rec_x2 = img_decoder(poe_joint_c)
        joint_gen_x2 = img_decoder(fixed_c)

        x1_samples += [
            rec_x1, cross_rec_x1, avg_joint_rec_x1, poe_joint_rec_x1, joint_gen_x1
        ]

        x2_samples += [
            rec_x2, cross_rec_x2, avg_joint_rec_x2, poe_joint_rec_x2, joint_gen_x2
        ]

    decoded_x1_samples = [cap_dataset.decode(s) for s in x1_samples]

    samples = ''
    filter_func = lambda w: w != '<pad>' and w != '<eos>'
    for tuples in zip(*decoded_x1_samples):
        print(len(tuples))
        fixed, rec, cross = tuples[:3]

        samples += ('input: ' + ' '.join(filter(filter_func, fixed)))
        samples += '<br><br>'

        samples += ('rec: ' + ' '.join(filter(filter_func, rec)))
        samples += '<br><br>'

        samples += ('cross: ' + ' '.join(filter(filter_func, cross)))
        samples += '<br><br>'

        if not opt.deterministic:
            joint_avg, joint_poe = tuples[3:-2]

            samples += ('joint_avg: ' + ' '.join(filter(filter_func, joint_avg)))
            samples += '<br><br>'

            samples += ('joint_poe: ' + ' '.join(filter(filter_func, joint_poe)))
            samples += '<br><br>'

        gen1, gen2 = tuples[-2:]

        samples += ('gen1: ' + ' '.join(filter(filter_func, gen1)))
        samples += '<br><br>'

        samples += ('gen2: ' + ' '.join(filter(filter_func, gen2)))
        samples += '<br><br>'

    writer.add_text('samples', samples, global_step=n_iter)

    model.train()


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


def eval_generation(n_iter):
    cmd = 'python {}/src/eval_cap_img.py'.format(output_dir)
    cmd += ' --dataroot {}'.format(opt.dataroot)
    cmd += ' --n_cpu {}'.format(opt.n_cpu)
    cmd += ' --batch_size {}'.format(opt.batch_size)
    cmd += ' --style_dim {}'.format(opt.style_dim)
    cmd += ' --latent_dim {}'.format(opt.latent_dim)
    cmd += ' --checkpoint_dir {}'.format(output_dir)
    cmd += ' --emb_size {}'.format(opt.emb_size)

    print('evaluating generation:', cmd)

    if opt.deterministic:
        acc_c2i, acc_i2c, acc_joint = subprocess.run(cmd,
                                                     capture_output=True, text=True, shell=True,
                                                     cwd='{}/src'.format(output_dir),
                                                     env=os.environ).stdout.strip().split('\n')[-3:]

        acc_c2i = float(acc_c2i)
        acc_i2c = float(acc_i2c)
        acc_joint = float(acc_joint)
        accuracies = {
            'm2s': acc_c2i,
            's2m': acc_i2c,
            'joint': acc_joint,
        }
    else:
        gt, acc_syn_c, acc_syn_i, acc_c2i, acc_i2c, acc_joint = subprocess.run(cmd,
                                                                               capture_output=True, text=True,
                                                                               shell=True,
                                                                               cwd='{}/src'.format(output_dir),
                                                                               env=os.environ).stdout.strip().split(
            '\n')[-6:]

        acc_syn_c = float(acc_syn_c)
        acc_syn_i = float(acc_syn_i)
        acc_c2i = float(acc_c2i)
        acc_i2c = float(acc_i2c)
        acc_joint = float(acc_joint)
        accuracies = {
            'groundtruth': gt,
            'syn_m': acc_syn_c,
            'syn_s': acc_syn_i,
            'm2s': acc_c2i,
            's2m': acc_i2c,
            'joint': acc_joint,
        }

    print(accuracies)

    writer.add_scalars('generation', accuracies, global_step=n_iter)


def main():
    x1_dataset = datasets.CUBCaptionVector(opt.dataroot, split='train', normalization='min-max')
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

    x1_discriminators = nn.ModuleList([
        models.cub_caption.XZDiscriminator(latent_dim=opt.latent_dim, emb_size=opt.emb_size),
        models.cub_caption.XZDiscriminator(latent_dim=opt.latent_dim, emb_size=opt.emb_size),
    ])
    x2_discriminators = nn.ModuleList([
        models.cub_image.XZDiscriminatorFT(latent_dim=opt.latent_dim),
        models.cub_image.XZDiscriminatorFT(latent_dim=opt.latent_dim),
    ])
    joint_discriminator = models.cub_caption_image.XXDiscriminatorFT(emb_size=opt.emb_size)

    model = models.mmali.FactorModelDoubleSemi(
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
            key_cap: x1_discriminators,
            key_img: x2_discriminators
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
                key_cap: {
                    'x': x1,
                    'z': torch.randn(opt.batch_size, opt.latent_dim).to(device),
                },
                key_img: {
                    'x': x2,
                    'z': torch.randn(opt.batch_size, opt.latent_dim).to(device)
                }
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

            model_ema.eval()

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

            save_samples(model_ema, fixed_x1, fixed_x2, fixed_s1, fixed_s2, fixed_c, n_iter, x1_dataset)
            save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G)
            model.train()

        if not opt.no_eval and n_iter > 0 and n_iter % opt.eval_interval == 0:
            try:
                eval_generation(n_iter)
            except:
                print('Something wrong during evaluation')

    if not opt.deterministic:
        paired_x1, paired_x2 = next(paired_dataloader)
        paired_x1 = paired_x1.to(device)
        paired_x2 = paired_x2.to(device)
        log_entropy(model_ema, paired_x1, paired_x2, n_iter)

    save_samples(model_ema, fixed_x1, fixed_x2, fixed_s1, fixed_s2, fixed_c, n_iter, x1_dataset)
    save_checkpoint(n_iter, model, model_ema, optimizer_D, optimizer_G)

    if not opt.no_eval:
        try:
            eval_generation(n_iter)
        except:
            print('Something wrong during evaluation')


if __name__ == '__main__':
    main()
