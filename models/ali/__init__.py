import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


def dis_dcgan(dis_fake, dis_real):
    loss = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    return loss


def gen_dcgan(dis_fake, dis_real=None):
    if dis_real is not None:
        return torch.mean(F.softplus(dis_real)) + torch.mean(F.softplus(-dis_fake))
    return torch.mean(F.softplus(-dis_fake))


def dis_kl(dis_fake, dis_real):
    loss = -torch.mean(dis_real) + torch.mean(torch.exp(dis_fake - 1.))
    return loss


def gen_kl(dis_fake, dis_real=None):
    if dis_real is not None:
        return torch.mean(dis_real) - torch.mean(torch.exp(dis_fake - 1.))
    return -torch.mean(torch.exp(dis_fake - 1.))


def dis_svae(dis_fake, dis_real):
    loss = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    return loss


def gen_svae(dis_fake, dis_real=None):
    if dis_real is not None:
        return torch.mean(dis_real) - torch.mean(dis_fake)
    return -torch.mean(dis_fake)


def dis_hinge(dis_fake, dis_real):
    loss = torch.mean(torch.relu(1. - dis_real)) + torch.mean(torch.relu(1. + dis_fake))
    return loss


def gen_hinge(dis_fake, dis_real=None):
    if dis_real is not None:
        return torch.mean(dis_real) - torch.mean(dis_fake)
    return -torch.mean(dis_fake)


class Model(nn.Module):
    def __init__(self, encoder, decoder, discriminator,
                 lambda_x_rec=0.0, lambda_z_rec=0.0, lambda_gp=(0.0, 0.0), dis_loss=dis_dcgan, gen_loss=gen_dcgan):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.lambda_gp = lambda_gp
        self.lambda_x_rec = lambda_x_rec
        self.lambda_z_rec = lambda_z_rec
        self.dis_loss = dis_loss
        self.gen_loss = gen_loss

    def lerp(self, other, beta):
        if beta == 1.0:
            return

        with torch.no_grad():
            params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            other_params = list(other.encoder.parameters()) + list(other.decoder.parameters())
            for p, p_other in zip(params, other_params):
                p.data.lerp_(p_other.data, 1.0 - beta)

            # TODO: how to handle batch norm buffer?
            buffers = list(self.encoder.buffers()) + list(self.decoder.buffers())
            other_buffers = list(other.encoder.buffers()) + list(other.decoder.buffers())
            for p, p_other in zip(buffers, other_buffers):
                p.data.copy_(p_other.data)

    def forward(self, x, z, train_d=True, progress=None):
        self.encoder.requires_grad_(not train_d)
        self.decoder.requires_grad_(not train_d)
        self.discriminator.requires_grad_(train_d)

        with torch.set_grad_enabled(not train_d):
            enc_z = self.encoder(x)
            dec_x = self.decoder(z)

        # if self.lambda_gp[0] > 0.0:
        #     x.requires_grad_(True)
        #     dec_x.requires_grad_(True)
        #     z.requires_grad_(True)
        #     enc_z.requires_grad_(True)

        dis_real = self.discriminator(x, enc_z)
        dis_fake = self.discriminator(dec_x, z)

        losses = {}
        if train_d:
            loss = self.dis_loss(dis_fake, dis_real)
            losses['dis'] = loss
            # if self.lambda_gp[0] > 0.0:
            #     gamma = self.lambda_gp[0] * self.lambda_gp[1] ** progress
            #     losses['gp'] = utils.calc_gradient_penalty_jsd(dis_real, [x, enc_z], dis_fake, [dec_x, z], gamma=gamma)
            #     losses['gp'] = utils.calc_gradient_penalty(self.discriminator, self.decoder, z, enc_z, lambda_gp=gamma)
        else:
            loss = self.gen_loss(dis_fake, dis_real)
            losses['gen'] = loss
            if self.lambda_x_rec > 0.0:
                rec_loss = (x - self.decoder(enc_z)).abs().mean()
                losses['x_rec'] = self.lambda_x_rec * rec_loss

            if self.lambda_z_rec > 0.0:
                rec_loss = (z - self.encoder(dec_x)).square().mean()
                losses['z_rec'] = self.lambda_z_rec * rec_loss

        return losses

    def forward1(self, x, z, train_d=True, progress=None):
        self.encoder.requires_grad_(not train_d)
        self.decoder.requires_grad_(not train_d)
        self.discriminator.requires_grad_(train_d)

        with torch.set_grad_enabled(not train_d):
            enc_z = self.encoder(x)
            dec_x = self.decoder(z)

        dis_1 = self.discriminator(x, enc_z)
        dis_2 = self.discriminator(dec_x, z)

        losses = {}
        if train_d:
            loss = self.dis_loss(dis_1, dis_2)
            losses['dis'] = loss
        else:
            loss = self.gen_loss(dis_1, dis_2)
            losses['gen'] = loss

        return losses


class PairModel(nn.Module):
    def __init__(self, encoder1, encoder2, decoder, discriminator,
                 style_dim=0,
                 lambda_x_rec=0.0, lambda_z_rec=0.0):
        super().__init__()

        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder = decoder
        self.discriminator = discriminator

        self.style_dim = style_dim
        self.lambda_x_rec = lambda_x_rec
        self.lambda_z_rec = lambda_z_rec

    def lerp(self, other, beta):
        if beta == 1.0:
            return

        with torch.no_grad():
            params = list(self.encoder1.parameters()) \
                     + list(self.encoder2.parameters()) \
                     + list(self.decoder.parameters())
            other_params = list(other.encoder1.parameters()) \
                           + list(other.encoder2.parameters()) \
                           + list(other.decoder.parameters())
            for p, p_other in zip(params, other_params):
                p.data.lerp_(p_other.data, 1.0 - beta)

            # TODO: how to handle batch norm buffer?
            buffers = list(self.encoder1.buffers()) \
                      + list(self.encoder2.buffers()) \
                      + list(self.decoder.buffers())
            other_buffers = list(other.encoder1.buffers()) \
                            + list(other.encoder2.buffers()) \
                            + list(other.decoder.buffers())
            for p, p_other in zip(buffers, other_buffers):
                p.data.copy_(p_other.data)

    # def forward(self, x1, x2, z, train_d=True, progress=None):
    #     self.encoder1.requires_grad_(not train_d)
    #     self.encoder2.requires_grad_(not train_d)
    #     self.decoder.requires_grad_(not train_d)
    #     self.discriminator.requires_grad_(train_d)
    #
    #     with torch.set_grad_enabled(not train_d):
    #         enc_z1 = self.encoder1(x1)
    #         enc_z2 = self.encoder2(x2)
    #         dec_x = self.decoder(z)
    #
    #     # dis_1 = self.discriminator(x1, enc_z1)
    #     dis_2 = self.discriminator(x1, torch.cat([
    #         enc_z1[:, :self.style_dim], enc_z2[:, self.style_dim:]
    #     ], dim=1))
    #     dis_3 = self.discriminator(dec_x, z)
    #
    #     losses = {}
    #     label_zeros = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)
    #     label_ones = torch.ones(x1.size(0), dtype=torch.long, device=x1.device)
    #     label_twos = 2 * torch.ones(x1.size(0), dtype=torch.long, device=x1.device)
    #     if train_d:
    #         loss = (
    #                    # F.cross_entropy(dis_1, label_zeros)
    #                        F.cross_entropy(dis_2, label_ones)
    #                        + F.cross_entropy(dis_3, label_twos)
    #                ) / 2
    #
    #         losses['dis'] = loss
    #     else:
    #         loss = (
    #                    # torch.mean(dis_1[:, 0] - dis_1[:, 1]) + torch.mean(dis_1[:, 0] - dis_1[:, 2])
    #                        torch.mean(dis_2[:, 1] - dis_2[:, 2]) + torch.mean(dis_2[:, 1] - dis_2[:, 0])
    #                        + torch.mean(dis_3[:, 2] - dis_3[:, 0]) + torch.mean(dis_3[:, 2] - dis_3[:, 1])
    #                ) / 2
    #         losses['gen'] = loss
    #
    #     return losses

    def forward(self, x1, x2, z, train_d=True, progress=None):
        self.encoder1.requires_grad_(not train_d)
        self.encoder2.requires_grad_(not train_d)
        self.decoder.requires_grad_(not train_d)
        self.discriminator.requires_grad_(train_d)

        with torch.set_grad_enabled(not train_d):
            enc_z1 = self.encoder1(x1)
            enc_z2 = self.encoder2(x2)
            dec_x = self.decoder(z)

        dis_1 = self.discriminator(x1, enc_z1)
        dis_2 = self.discriminator(x1, torch.cat([
            enc_z1[:, :self.style_dim], enc_z2[:, self.style_dim:]
        ], dim=1))
        dis_3 = self.discriminator(dec_x, z)

        losses = {}
        label_zeros = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)
        label_ones = torch.ones(x1.size(0), dtype=torch.long, device=x1.device)
        label_twos = 2 * torch.ones(x1.size(0), dtype=torch.long, device=x1.device)
        if train_d:
            loss = (
                           F.cross_entropy(dis_1, label_zeros)
                           + F.cross_entropy(dis_2, label_ones)
                           + F.cross_entropy(dis_3, label_twos)
                   ) / 3
            losses['dis'] = loss
        else:
            # loss = (
            #                F.cross_entropy(dis_1, label_ones) + F.cross_entropy(dis_1, label_twos)
            #                + F.cross_entropy(dis_2, label_twos) + F.cross_entropy(dis_2, label_zeros)
            #                + F.cross_entropy(dis_3, label_zeros) + F.cross_entropy(dis_3, label_ones)
            #        ) / 6
            loss = (
                           torch.mean(dis_1[:, 0] - dis_1[:, 1]) + torch.mean(dis_1[:, 0] - dis_1[:, 2])
                           + torch.mean(dis_2[:, 1] - dis_2[:, 2]) + torch.mean(dis_2[:, 1] - dis_2[:, 0])
                           + torch.mean(dis_3[:, 2] - dis_3[:, 0]) + torch.mean(dis_3[:, 2] - dis_3[:, 1])
                   ) / 3
            losses['gen'] = loss

            # rec_x1 = self.decoder(enc_z1)
            # loss = (x1 - rec_x1).square().mean()
            # losses['x_rec'] = loss

        return losses
