import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class FactorModel(nn.Module):
    def __init__(self, encoders, decoders, xz_discriminators, joint_discriminator,
                 content_dim=20, lambda_unimodal=1.0, lambda_x_rec=0.0, lambda_c_rec=0.0, lambda_s_rec=0.0):
        super().__init__()

        assert len(encoders.items()) == len(decoders.items())
        assert len(encoders.items()) == len(xz_discriminators.items())

        self.encoders = nn.ModuleList()
        for name, module in encoders.items():
            self.encoders.add_module(name, module)

        self.decoders = nn.ModuleList()
        for name, module in decoders.items():
            self.decoders.add_module(name, module)

        self.xz_discriminators = nn.ModuleList()
        for name, module in xz_discriminators.items():
            self.xz_discriminators.add_module(name, module)

        self.joint_discriminator = joint_discriminator

        self.content_dim = content_dim
        self.lambda_unimodal = lambda_unimodal
        self.lambda_x_rec = lambda_x_rec
        self.lambda_c_rec = lambda_c_rec
        self.lambda_s_rec = lambda_s_rec

        self.n_modalities = len(encoders.items())

        self.generators = nn.ModuleList()
        self.generators.add_module('encoders', self.encoders)
        self.generators.add_module('decoders', self.decoders)

        self.discriminators = nn.ModuleList()
        self.discriminators.add_module('xz_discriminators', self.xz_discriminators)
        self.discriminators.add_module('joint_discriminator', self.joint_discriminator)

        self.sorted_keys = sorted(encoders.keys())

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

    def calc_joint_score(self, inputs, score_joint):
        scores = {}
        for modality_key in self.sorted_keys:
            discriminator = getattr(self.xz_discriminators, modality_key)
            x = inputs[modality_key]['x']
            z = inputs[modality_key]['z']

            scores[modality_key] = discriminator(x, z)

        score_sum = score_joint + torch.sum(torch.stack([s[:, -1] - s[:, 1]  # q(x, s) p(c) : p(x, s, c)
                                                         for s in scores.values()], dim=0), dim=0)

        joint_score = []
        for modality_key in self.sorted_keys:
            s = scores[modality_key]
            # q(x, s, c) : q(x, s) p(c)
            # score = score_sum + (s[:, 0] - s[:, -1])
            score = s[:, 0] - s[:, -1]
            joint_score.append(score)

        joint_score.append(-score_sum)
        return torch.stack(joint_score, dim=1)

    def forward(self, real_inputs, train_d=True, joint=False, progress=None):
        return self.forward_jsd(real_inputs, train_d=train_d, joint=joint, progress=progress)

    def forward_jsd(self, real_inputs, train_d=True, joint=False, progress=None):
        self.generators.requires_grad_(not train_d)
        self.discriminators.requires_grad_(train_d)

        batch_size = list(real_inputs.values())[0]['x'].size(0)
        device = list(real_inputs.values())[0]['x'].device
        label_zeros = torch.zeros(batch_size, dtype=torch.long, device=device)
        label_ones = torch.ones(batch_size, dtype=torch.long, device=device)

        losses = {}
        if train_d:
            if joint:
                gen_inputs = {}
                with torch.set_grad_enabled(False):
                    for modality_key in self.sorted_keys:
                        encoder = getattr(self.encoders, modality_key)

                        real_x = real_inputs[modality_key]['x']

                        enc_z = encoder(real_x)

                        gen_inputs[modality_key] = {'z': enc_z}

                # make sure the order of sorted keys matches the input order of joint discriminator
                joint_inputs = [real_inputs[k]['x'] for k in self.sorted_keys]
                shuffled_inputs = [real_inputs[k]['extra_x'] for k in self.sorted_keys]

                dis_real = self.joint_discriminator(*joint_inputs)
                dis_fake = self.joint_discriminator(*shuffled_inputs)
                losses['joint'] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))

                for modality_key in self.sorted_keys:
                    discriminator = getattr(self.xz_discriminators, modality_key)
                    real_x = real_inputs[modality_key]['x']
                    enc_z = gen_inputs[modality_key]['z']

                    enc_s = enc_z[:, :-self.content_dim]

                    # for other keys
                    label_value = 2
                    for other_k in self.sorted_keys:
                        # q_j(x_i, s_i, c) where j != i
                        if other_k == modality_key:
                            continue

                        other_enc_z = gen_inputs[other_k]['z']
                        other_enc_c = other_enc_z[:, -self.content_dim:]

                        dis_other = discriminator(real_x, torch.cat([enc_s, other_enc_c], dim=1))
                        losses['{}_c{}'.format(modality_key, label_value)] = F.cross_entropy(dis_other,
                                                                                             label_value * label_ones)

                        label_value += 1
            else:
                gen_inputs = {}
                with torch.set_grad_enabled(False):
                    for modality_key in self.sorted_keys:
                        encoder = getattr(self.encoders, modality_key)
                        decoder = getattr(self.decoders, modality_key)

                        real_x = real_inputs[modality_key]['x']
                        real_z = real_inputs[modality_key]['z']

                        enc_z = encoder(real_x)
                        dec_x = decoder(real_z)

                        gen_inputs[modality_key] = {'x': dec_x, 'z': enc_z}

                for modality_key in self.sorted_keys:
                    discriminator = getattr(self.xz_discriminators, modality_key)

                    real_x = real_inputs[modality_key]['x']
                    enc_z = gen_inputs[modality_key]['z']

                    dec_x = gen_inputs[modality_key]['x']
                    real_z = real_inputs[modality_key]['z']

                    # shuffle x and z together
                    real_x_shuffled, enc_z_shuffled = utils.permute_dim([real_x, enc_z], dim=0)
                    enc_s_shuffled = enc_z_shuffled[:, :-self.content_dim]

                    real_z_shuffled = utils.permute_dim(real_z, dim=0)
                    real_c_shuffled = real_z_shuffled[:, -self.content_dim:]

                    cs_shuffled = torch.cat([enc_s_shuffled, real_c_shuffled], dim=1)

                    # q(x, s, c)
                    losses['{}_c0'.format(modality_key)] = F.cross_entropy(discriminator(real_x, enc_z), label_zeros)

                    # p(x, s, c)
                    losses['{}_c1'.format(modality_key)] = F.cross_entropy(discriminator(dec_x, real_z), label_ones)

                    # q(x, s) p(c)
                    losses['{}_c{}'.format(modality_key, 1 + self.n_modalities)] = F.cross_entropy(
                        discriminator(real_x_shuffled, cs_shuffled), (1 + self.n_modalities) * label_ones)
        else:
            gen_inputs = {}
            with torch.set_grad_enabled(True):
                for modality_key in self.sorted_keys:
                    encoder = getattr(self.encoders, modality_key).module
                    decoder = getattr(self.decoders, modality_key)

                    real_x = real_inputs[modality_key]['x']
                    real_z = real_inputs[modality_key]['z']

                    enc_z_dist_param = encoder(real_x)
                    enc_z = utils.reparameterize(enc_z_dist_param)
                    dec_x = decoder(real_z)

                    gen_inputs[modality_key] = {'x': dec_x, 'z': enc_z, 'z_dist_param': enc_z_dist_param}

            if joint:
                label_value = 0
                # encoder distributions
                score_q = self.joint_discriminator(*[real_inputs[k]['x'] for k in self.sorted_keys])[:, 0]
                for modality_key in self.sorted_keys:
                    curr_inputs = {k: {} for k in self.sorted_keys}
                    enc_c = gen_inputs[modality_key]['z'][:, -self.content_dim:]

                    # use k2's style code + k's content code
                    for modality_key2 in self.sorted_keys:
                        curr_inputs[modality_key2]['x'] = real_inputs[modality_key2]['x']

                        enc_s = gen_inputs[modality_key2]['z'][:, :-self.content_dim]
                        curr_inputs[modality_key2]['z'] = torch.cat([enc_s, enc_c], dim=1)

                    dis_score = self.calc_joint_score(curr_inputs, score_q)
                    adv_losses = [F.cross_entropy(dis_score, i * label_ones)
                                  for i in range(dis_score.size(1)) if i != label_value]
                    losses['joint_q{}'.format(label_value)] = 2. / (1 + self.n_modalities) * torch.mean(
                        torch.stack(adv_losses, dim=0), dim=0)

                    label_value += 1

                # decoder distribution
                score_p = self.joint_discriminator(*[gen_inputs[k]['x'] for k in self.sorted_keys])[:, 0]
                curr_inputs = {k: {} for k in self.sorted_keys}
                for modality_key in self.sorted_keys:
                    curr_inputs[modality_key]['x'] = gen_inputs[modality_key]['x']
                    curr_inputs[modality_key]['z'] = real_inputs[modality_key]['z']

                dis_score = self.calc_joint_score(curr_inputs, score_p)
                adv_losses = [F.cross_entropy(dis_score, i * label_ones)
                              for i in range(dis_score.size(1)) if i != label_value]
                losses['joint_p{}'.format(label_value)] = 2. / (1 + self.n_modalities) * torch.mean(
                    torch.stack(adv_losses, dim=0), dim=0)
                label_value += 1

                for modality_key in self.sorted_keys:
                    discriminator = getattr(self.xz_discriminators, modality_key)
                    real_x = real_inputs[modality_key]['x']
                    enc_z = gen_inputs[modality_key]['z']

                    enc_s = enc_z[:, :-self.content_dim]

                    # for other keys
                    label_value = 2
                    for other_k in self.sorted_keys:
                        # q_j(x_i, s_i, c) where j != i
                        if other_k == modality_key:
                            continue

                        other_enc_c = gen_inputs[other_k]['z'][:, -self.content_dim:]
                        z_combined = torch.cat([enc_s, other_enc_c], dim=1)

                        dis_other = discriminator(real_x, z_combined)
                        dis_other = dis_other[:, :-1]

                        adv_losses = [F.cross_entropy(dis_other, i * label_ones)
                                      for i in range(dis_other.size(1)) if i != label_value]

                        losses['{}_c{}'.format(modality_key, label_value)] = self.lambda_unimodal * 2. / (
                                1 + self.n_modalities) * torch.mean(
                            torch.stack(adv_losses, dim=0), dim=0)

                        label_value += 1

                if self.lambda_c_rec > 0.0:
                    z_rec_dist_params = []

                    for modality_key in self.sorted_keys:
                        encoder = getattr(self.encoders, modality_key).module
                        z_rec_dist_param = encoder(gen_inputs[modality_key]['x'])
                        z_rec_dist_params.append(z_rec_dist_param)

                    z_joint = utils.reparameterize(utils.joint_posterior(*z_rec_dist_params, average=True))
                    c_joint_rec = z_joint[:, -self.content_dim:]

                    c_real = list(real_inputs.values())[0]['z'][:, -self.content_dim:]
                    c_rec_loss = (c_joint_rec - c_real).square().mean()
                    losses['joint_c_rec'] = self.lambda_c_rec * c_rec_loss

                if self.lambda_x_rec > 0.0:
                    z_joint = utils.reparameterize(
                        utils.joint_posterior(*[v['z_dist_param'] for v in gen_inputs.values()], average=True))

                    c_joint = z_joint[:, -self.content_dim:]

                    for modality_key in self.sorted_keys:
                        decoder = getattr(self.decoders, modality_key)

                        x_real = real_inputs[modality_key]['x']
                        x_rec = decoder(torch.cat([gen_inputs[modality_key]['z'][:, :-self.content_dim], c_joint],
                                                  dim=1))

                        x_rec_loss = (x_rec - x_real).square().mean()
                        losses['{}_x_rec_joint'.format(modality_key)] = self.lambda_x_rec * x_rec_loss
            else:
                for modality_key in self.sorted_keys:
                    discriminator = getattr(self.xz_discriminators, modality_key)

                    real_x = real_inputs[modality_key]['x']
                    enc_z = gen_inputs[modality_key]['z']

                    dec_x = gen_inputs[modality_key]['x']
                    real_z = real_inputs[modality_key]['z']

                    # q(x, s, c)
                    dis_0 = discriminator(real_x, enc_z)
                    dis_0 = dis_0[:, :-1]
                    label_value = 0
                    adv_losses = [F.cross_entropy(dis_0, i * label_ones)
                                  for i in range(dis_0.size(1)) if i != label_value]

                    losses['{}_c0'.format(modality_key)] = self.lambda_unimodal * 2. / (
                                1 + self.n_modalities) * torch.mean(torch.stack(adv_losses,
                                                                                dim=0), dim=0)

                    # p(x, s, c)
                    dis_1 = discriminator(dec_x, real_z)
                    dis_1 = dis_1[:, :-1]
                    label_value = 1
                    adv_losses = [F.cross_entropy(dis_1, i * label_ones)
                                  for i in range(dis_1.size(1)) if i != label_value]
                    losses['{}_c1'.format(modality_key)] = self.lambda_unimodal * 2. / (
                                1 + self.n_modalities) * torch.mean(torch.stack(adv_losses,
                                                                                dim=0), dim=0)

                if self.lambda_s_rec > 0.0:
                    for modality_key in self.sorted_keys:
                        encoder = getattr(self.encoders, modality_key)
                        z_rec = encoder(gen_inputs[modality_key]['x'])

                        s_real = real_inputs[modality_key]['z'][:, :-self.content_dim]
                        s_rec = z_rec[:, :-self.content_dim]

                        s_rec_loss = (s_rec - s_real).square().mean()
                        losses['{}_s_rec'.format(modality_key)] = self.lambda_s_rec * s_rec_loss

        return losses
