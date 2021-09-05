import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class JointModel(nn.Module):
    def __init__(self, encoders, decoders, discriminator,
                 content_dim=20, lambda_x_rec=0.0, lambda_c_rec=0.0, lambda_s_rec=0.0, mod_coeff=None,
                 joint_rec=False):
        super().__init__()

        self.encoders = nn.ModuleList()
        for name, module in encoders.items():
            self.encoders.add_module(name, module)

        self.decoders = nn.ModuleList()
        for name, module in decoders.items():
            self.decoders.add_module(name, module)

        self.discriminators = discriminator
        self.content_dim = content_dim
        self.lambda_x_rec = lambda_x_rec
        self.lambda_c_rec = lambda_c_rec
        self.lambda_s_rec = lambda_s_rec
        self.joint_rec = joint_rec
        self.mod_coeff = mod_coeff if mod_coeff else {k: 1.0 for k in encoders.keys()}
        self.n_modalities = len(encoders)

        self.generators = nn.ModuleList()
        self.generators.add_module('encoders', self.encoders)
        self.generators.add_module('decoders', self.decoders)

        self.sorted_keys = sorted(encoders.keys())

    # def __init__(self, encoders, decoders, discriminator,
    #              joint_posterior=False, style_dim=0, content_dim=20, lambda_x_rec=0.0):
    #     super().__init__()
    #     assert len(encoders) == len(decoders)
    #
    #     self.encoders = nn.ModuleList(encoders)
    #     self.decoders = nn.ModuleList(decoders)
    #     self.discriminator = discriminator
    #     self.joint_posterior = joint_posterior
    #     self.style_dim = style_dim
    #     self.content_dim = content_dim
    #     self.lambda_x_rec = lambda_x_rec
    #     self.n_modalities = len(encoders)
    #
    #     self.generators = nn.ModuleList()
    #     self.generators.add_module('encoders', self.encoders)
    #     self.generators.add_module('decoders', self.decoders)

    @torch.no_grad()
    def lerp(self, other, beta):
        if beta == 1.0:
            return

        params = list(self.generators.parameters())
        other_params = list(other.generators.parameters())
        for p, p_other in zip(params, other_params):
            p.data.lerp_(p_other.data, 1.0 - beta)

        # TODO: to handle batch norm buffer?
        buffers = list(self.generators.buffers())
        other_buffers = list(other.generators.buffers())
        for p, p_other in zip(buffers, other_buffers):
            p.data.copy_(p_other.data)

    def encode(self, *X, no_joint=False, no_sampling=False):
        Z = [encoder(X[i]) for i, encoder in enumerate(self.encoders)]
        if self.joint_posterior and not no_joint:
            power_Z = utils.powerset(Z, remove_null=True)
            Z = []
            for sub_Z in power_Z:
                mu = []
                logvar = []
                for z in sub_Z:
                    # split mu, logvar
                    mu.append(z[:, :z.size(1) // 2])
                    logvar.append(z[:, z.size(1) // 2:])
                Z.append(utils.product_of_experts(torch.stack(mu, dim=0),
                                                  torch.stack(logvar, dim=0),
                                                  average=True))
        if no_sampling:
            return Z
        return [utils.reparameterize(z) for z in Z]

    def decode(self, z):
        dec_X = []
        for i, decoder in enumerate(self.decoders):
            # z is composed of style_1, style_2, ..., style_N, content
            # combine style_i and content and decode
            # if style_dim == 0, every z has the same value
            input_z = torch.cat([z[:, i * self.style_dim:(i + 1) * self.style_dim],
                                 z[:, -self.content_dim:]], dim=1)
            dec_X.append(decoder(input_z))
        return dec_X

    def joint_encode(self, *X, no_sampling=False):
        mu = []
        logvar = []
        for i, encoder in enumerate(self.encoders):
            z = encoder(X[i])

            mu.append(z[:, :z.size(1) // 2])
            logvar.append(z[:, z.size(1) // 2:])

        poe_z = utils.product_of_experts(torch.stack(mu, dim=0),
                                         torch.stack(logvar, dim=0),
                                         average=True)
        if no_sampling:
            return poe_z
        return utils.reparameterize(poe_z)

    def forward(self, real_inputs, train_d=True, joint=False, progress=None):
        return self.forward_jsd(real_inputs, train_d=train_d, progress=progress)

    def forward1(self, z, X, train_d=True):
        self.encoders.requires_grad_(not train_d)
        self.decoders.requires_grad_(not train_d)
        self.discriminators.requires_grad_(train_d)

        label_ones = torch.ones(X[0].size(0), dtype=torch.long, device=X[0].device)

        with torch.set_grad_enabled(not train_d):
            enc_Z = self.encode(*X)
            dec_X = self.decode(z)

            real_X = X
            enc_style_z = torch.cat([enc_z[:, :-self.content_dim] for enc_z in enc_Z[:self.n_modalities]], dim=1)

        scores = []
        labels = []
        for enc_z in enc_Z:
            score = self.discriminators(*real_X, torch.cat([enc_style_z, enc_z[:, -self.content_dim:]], dim=1))
            scores.append(score)
            labels.append(len(labels) * label_ones)

        score = self.discriminators(*dec_X, z)
        scores.append(score)
        labels.append(len(labels) * label_ones)

        scores = torch.cat(scores, dim=0)
        if train_d:
            return {'adv': F.cross_entropy(scores, torch.cat(labels, dim=0))}
        else:
            adv_losses = []
            for i in range(len(labels) - 1):
                # shift labels
                labels.insert(0, labels.pop())
                adv_losses.append(F.cross_entropy(scores, torch.cat(labels, dim=0)))

            adv_losses = torch.sum(torch.stack(adv_losses, dim=0), dim=0)

            # JSD
            if self.lambda_x_rec > 0.:
                rec_losses = [(X[i] - decoder(enc_Z[i])).square().mean() for i, decoder in enumerate(self.decoders)]
                rec_losses = self.lambda_x_rec * torch.sum(torch.stack(rec_losses, dim=0), dim=0)

                return {'adv': adv_losses, 'rec': rec_losses}

            return {'adv': adv_losses}

    def forward2(self, z, X, train_d=True):
        self.encoders.requires_grad_(not train_d)
        self.decoders.requires_grad_(not train_d)
        self.discriminators.requires_grad_(train_d)

        with torch.set_grad_enabled(not train_d):
            enc_z = self.joint_encode(*X, no_sampling=False)
            dec_X = self.decode(z)

        dis_real = self.discriminators(*X, enc_z)
        dis_fake = self.discriminators(*dec_X, z)

        losses = {}
        if train_d:
            # loss = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
            loss = torch.mean(torch.relu(1. - dis_real)) + torch.mean(torch.relu(1. + dis_fake))
            losses['dis'] = loss

        else:
            # loss = torch.mean(F.softplus(dis_real)) + torch.mean(F.softplus(-dis_fake))
            loss = torch.mean(dis_real) - torch.mean(dis_fake)
            losses['gen'] = loss

        return losses

    def forward_jsd(self, real_inputs, train_d=True, progress=None):
        self.generators.requires_grad_(not train_d)
        self.discriminators.requires_grad_(train_d)

        batch_size = list(real_inputs.values())[0]['x'].size(0)
        device = list(real_inputs.values())[0]['x'].device
        label_ones = torch.ones(batch_size, dtype=torch.long, device=device)

        gen_inputs = {}
        with torch.set_grad_enabled(not train_d):
            for modality_key in self.sorted_keys:
                encoder = getattr(self.encoders, modality_key)
                decoder = getattr(self.decoders, modality_key)

                real_x = real_inputs[modality_key]['x']
                real_z = real_inputs[modality_key]['z']

                enc_z = encoder(real_x)
                dec_x = decoder(real_z)

                gen_inputs[modality_key] = {'x': dec_x, 'z': enc_z}

        scores = []
        for modality_key in self.sorted_keys:
            curr_x = [real_inputs[k]['x'] for k in self.sorted_keys]
            styles = [gen_inputs[k]['z'][:, :-self.content_dim] for k in self.sorted_keys]
            content = gen_inputs[modality_key]['z'][:, -self.content_dim:]
            z = torch.cat(styles + [content], dim=1)
            curr_inputs = curr_x + [z]

            score = self.discriminators(*curr_inputs)
            scores.append(score)

        curr_x = [gen_inputs[k]['x'] for k in self.sorted_keys]
        styles = [real_inputs[k]['z'][:, :-self.content_dim] for k in self.sorted_keys]
        content = list(real_inputs.values())[0]['z'][:, -self.content_dim:]
        z = torch.cat(styles + [content], dim=1)
        curr_inputs = curr_x + [z]
        score = self.discriminators(*curr_inputs)
        scores.append(score)

        losses = {}
        if train_d:
            for label_value, score in enumerate(scores):
                losses['{}_c{}'.format(modality_key, label_value)] = F.cross_entropy(score, label_value * label_ones)
        else:
            for label_value, score in enumerate(scores):
                adv_losses = [F.cross_entropy(score, i * label_ones) for i in range(score.size(1)) if i != label_value]
                losses['{}_c{}'.format(modality_key, label_value)] = 2. / (1 + self.n_modalities) * torch.mean(
                    torch.stack(adv_losses, dim=0), dim=0)

        return losses
