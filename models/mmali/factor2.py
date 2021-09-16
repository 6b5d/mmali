import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class FactorModelDoubleSemi(nn.Module):
    def __init__(self, encoders, decoders, xz_discriminators, joint_discriminator,
                 content_dim=20, lambda_unimodal=1.0, lambda_x_rec=0.0, lambda_c_rec=0.0, lambda_s_rec=0.0,
                 mod_coeff=None, joint_rec=False):
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
        self.joint_rec = joint_rec
        self.mod_coeff = mod_coeff if mod_coeff else {k: 1.0 for k in encoders.keys()}

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

    def forward(self, real_inputs, train_d=True, joint=False, progress=None):
        return self.forward_jsd(real_inputs, train_d=train_d, joint=joint, progress=progress)

    def calc_joint_score_jsd(self, inputs, score_joint):
        scores = {}
        for modality_key in self.sorted_keys:
            discriminator = getattr(self.xz_discriminators, modality_key)
            x = inputs[modality_key]['x']
            z = inputs[modality_key]['z']

            score1 = discriminator[0](x, z)  # q(x, s, c) : p(x, s, c)
            score2 = discriminator[1](x, z)  # q(x, s, c) : q(x, s) p(c)

            scores[modality_key] = [score1, score2]

        score_sum = score_joint + torch.sum(torch.stack([s[0] - s[1]  # q(x, s) p(c) : p(x, s, c)
                                                         for s in scores.values()], dim=0), dim=0)

        joint_score = []
        for modality_key in self.sorted_keys:
            # q(x, s, c) : q(x, s) p(c)
            score = scores[modality_key][1]
            joint_score.append(score)

        joint_score.append(-score_sum)

        # for modality_key in self.sorted_keys:
        #     # q(x, s, c) : p(x, s, c)
        #     score = scores[modality_key][0]
        #     joint_score.append(score - score_sum)

        return torch.cat(joint_score, dim=1)

    def forward_jsd(self, real_inputs, train_d=True, joint=False, progress=None):
        self.generators.requires_grad_(not train_d)
        self.discriminators.requires_grad_(train_d)

        batch_size = list(real_inputs.values())[0]['x'].size(0)
        device = list(real_inputs.values())[0]['x'].device
        label_ones = torch.ones(batch_size, dtype=torch.long, device=device)

        losses = {}
        if train_d:
            if joint:
                joint_inputs = [real_inputs[k]['x'] for k in self.sorted_keys]
                shuffled_inputs = [real_inputs[k]['extra_x'] for k in self.sorted_keys]

                dis_real = self.joint_discriminator(*joint_inputs)
                dis_fake = self.joint_discriminator(*shuffled_inputs)

                losses['joint'] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
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

                    # q(x, s, c) : p(x, s, c)
                    dis_real = discriminator[0](real_x, enc_z)
                    dis_fake = discriminator[0](dec_x, real_z)
                    losses['{}_0'.format(modality_key)] = torch.mean(F.softplus(-dis_real)) + \
                                                          torch.mean(F.softplus(dis_fake))

                    # shuffle x and z together
                    real_x_shuffled, enc_z_shuffled = utils.permute_dim([real_x, enc_z], dim=0)
                    enc_s_shuffled = enc_z_shuffled[:, :-self.content_dim]
                    real_z_shuffled = utils.permute_dim(real_z, dim=0)
                    real_c_shuffled = real_z_shuffled[:, -self.content_dim:]
                    cs_shuffled = torch.cat([enc_s_shuffled, real_c_shuffled], dim=1)

                    # q(x, s, c) : q(x, s) q(c)
                    dis_real = discriminator[1](real_x, enc_z)
                    dis_fake = discriminator[1](real_x_shuffled, cs_shuffled)

                    losses['{}_1'.format(modality_key)] = torch.mean(F.softplus(-dis_real)) + \
                                                          torch.mean(F.softplus(dis_fake))
        else:
            gen_inputs = {}
            with torch.set_grad_enabled(True):
                for modality_key in self.sorted_keys:
                    if self.joint_rec:
                        encoder = getattr(self.encoders, modality_key).module
                        decoder = getattr(self.decoders, modality_key)

                        real_x = real_inputs[modality_key]['x']
                        real_z = real_inputs[modality_key]['z']

                        enc_z_dist_param = encoder(real_x)
                        enc_z = utils.reparameterize(enc_z_dist_param)
                        dec_x = decoder(real_z)

                        gen_inputs[modality_key] = {'x': dec_x, 'z': enc_z, 'z_dist_param': enc_z_dist_param}
                    else:
                        encoder = getattr(self.encoders, modality_key)
                        decoder = getattr(self.decoders, modality_key)

                        real_x = real_inputs[modality_key]['x']
                        real_z = real_inputs[modality_key]['z']

                        enc_z = encoder(real_x)
                        dec_x = decoder(real_z)

                        gen_inputs[modality_key] = {'x': dec_x, 'z': enc_z}

            if joint:
                label_value = 0
                score_q = self.joint_discriminator(*[real_inputs[k]['x'] for k in self.sorted_keys])
                for modality_key in self.sorted_keys:
                    curr_inputs = {k: {} for k in self.sorted_keys}
                    content = gen_inputs[modality_key]['z'][:, -self.content_dim:]

                    for modality_key2 in self.sorted_keys:
                        style = gen_inputs[modality_key2]['z'][:, :-self.content_dim]

                        curr_inputs[modality_key2]['x'] = real_inputs[modality_key2]['x']
                        curr_inputs[modality_key2]['z'] = torch.cat([style, content], dim=1)

                    dis_score = self.calc_joint_score_jsd(curr_inputs, score_q)
                    adv_losses = [F.cross_entropy(dis_score, i * label_ones)
                                  for i in range(dis_score.size(1)) if i != label_value]
                    losses['joint_q{}'.format(label_value)] = \
                        2. / (1 + self.n_modalities) * torch.mean(torch.stack(adv_losses, dim=0), dim=0)

                    label_value += 1

                score_p = self.joint_discriminator(*[gen_inputs[k]['x'] for k in self.sorted_keys])
                curr_inputs = {k: {} for k in self.sorted_keys}
                for modality_key in self.sorted_keys:
                    curr_inputs[modality_key]['x'] = gen_inputs[modality_key]['x']
                    curr_inputs[modality_key]['z'] = real_inputs[modality_key]['z']

                dis_score = self.calc_joint_score_jsd(curr_inputs, score_p)
                adv_losses = [F.cross_entropy(dis_score, i * label_ones)
                              for i in range(dis_score.size(1)) if i != label_value]
                losses['joint_p{}'.format(label_value)] = \
                    2. / (1 + self.n_modalities) * torch.mean(torch.stack(adv_losses, dim=0), dim=0)

                label_value += 1

                if self.joint_rec:
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
                            losses['{}_x_rec_joint'.format(modality_key)] = \
                                self.mod_coeff[modality_key] * self.lambda_x_rec * x_rec_loss

            else:
                # label_value = self.n_modalities + 1
                # for modality_key in self.sorted_keys:
                #     curr_inputs = {k: {} for k in self.sorted_keys}
                #     curr_inputs[modality_key]['x'] = real_inputs[modality_key]['x']
                #     curr_inputs[modality_key]['z'] = gen_inputs[modality_key]['z']
                #
                #     enc_c = gen_inputs[modality_key]['z'][:, -self.content_dim:]
                #     for other_key in self.sorted_keys:
                #         if other_key == modality_key:
                #             continue
                #
                #         decoder = getattr(self.decoders, other_key)
                #         real_s = utils.permute_dim(real_inputs[other_key]['z'][:, :-self.content_dim], dim=0)
                #         combined_z = torch.cat([real_s, enc_c], dim=1)
                #
                #         rec_x = decoder(combined_z)
                #         curr_inputs[other_key]['x'] = rec_x
                #         curr_inputs[other_key]['z'] = combined_z
                #
                #     score_joint = self.joint_discriminator(*[curr_inputs[k]['x'] for k in self.sorted_keys])
                #     dis_score = self.calc_joint_score(curr_inputs, score_joint)
                #
                #     adv_losses = [F.cross_entropy(dis_score, i * label_ones)
                #                   for i in range(dis_score.size(1)) if i != label_value]
                #
                #     losses['joint_r{}'.format(label_value)] = \
                #         2. / (1 + 2 * self.n_modalities) * torch.mean(torch.stack(adv_losses, dim=0), dim=0)
                #
                #     label_value += 1

                for modality_key in self.sorted_keys:
                    discriminator = getattr(self.xz_discriminators, modality_key)[0]

                    real_x = real_inputs[modality_key]['x']
                    enc_z = gen_inputs[modality_key]['z']

                    dec_x = gen_inputs[modality_key]['x']
                    real_z = real_inputs[modality_key]['z']

                    dis_real = discriminator(real_x, enc_z)
                    dis_fake = discriminator(dec_x, real_z)
                    loss = torch.mean(F.softplus(dis_real)) + torch.mean(F.softplus(-dis_fake))
                    losses['{}_real_fake'.format(modality_key)] = self.lambda_unimodal * loss

                if not self.joint_rec:
                    if self.lambda_x_rec > 0.0:
                        for modality_key in self.sorted_keys:
                            decoder = getattr(self.decoders, modality_key)
                            x_real = real_inputs[modality_key]['x']
                            enc_z = gen_inputs[modality_key]['z']

                            x_rec = decoder(enc_z)

                            x_rec_loss = (x_rec - x_real).square().mean()

                            losses['{}_x_rec'.format(modality_key)] = \
                                self.mod_coeff[modality_key] * self.lambda_x_rec * x_rec_loss

                if (self.lambda_c_rec > 0.0 and not self.joint_rec) or self.lambda_s_rec > 0.0:
                    for modality_key in self.sorted_keys:
                        encoder = getattr(self.encoders, modality_key)
                        z_rec = encoder(gen_inputs[modality_key]['x'])

                        if self.lambda_c_rec > 0.0 and not self.joint_rec:
                            c_real = real_inputs[modality_key]['z'][:, -self.content_dim:]
                            c_rec = z_rec[:, -self.content_dim:]
                            c_rec_loss = (c_rec - c_real).square().mean()
                            losses['{}_c_rec'.format(modality_key)] = self.lambda_c_rec * c_rec_loss

                        if self.lambda_s_rec > 0.0:
                            s_real = real_inputs[modality_key]['z'][:, :-self.content_dim]
                            s_rec = z_rec[:, :-self.content_dim]
                            s_rec_loss = (s_rec - s_real).square().mean()
                            losses['{}_s_rec'.format(modality_key)] = self.lambda_s_rec * s_rec_loss

        return losses

    def calc_joint_score_jsd2(self, inputs, score_joint):
        scores = {}
        for modality_key in self.sorted_keys:
            discriminator = getattr(self.xz_discriminators, modality_key)
            x = inputs[modality_key]['x']
            z = inputs[modality_key]['z']

            score1 = discriminator[0](x, z)
            score1 = (score1[:, 0] - score1[:, 1]).unsqueeze(dim=-1)  # q(x, s, c) : p(x, s, c)
            score2 = discriminator[1](x, z)  # q(x, s, c) : q(x, s) p(c)

            scores[modality_key] = [score1, score2]

        score_sum = score_joint + torch.sum(torch.stack([s[0] - s[1]  # q(x, s) p(c) : p(x, s, c)
                                                         for s in scores.values()], dim=0), dim=0)

        joint_score = []
        for modality_key in self.sorted_keys:
            # q(x, s, c) : q(x, s) p(c)
            score = scores[modality_key][1]
            joint_score.append(score)

        joint_score.append(-score_sum)

        return torch.cat(joint_score, dim=1)

    def forward_jsd2(self, real_inputs, train_d=True, joint=False, progress=None):
        self.generators.requires_grad_(not train_d)
        self.discriminators.requires_grad_(train_d)

        batch_size = list(real_inputs.values())[0]['x'].size(0)
        device = list(real_inputs.values())[0]['x'].device
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

                joint_inputs = [real_inputs[k]['x'] for k in self.sorted_keys]
                shuffled_inputs = [real_inputs[k]['extra_x'] for k in self.sorted_keys]

                dis_real = self.joint_discriminator(*joint_inputs)
                dis_fake = self.joint_discriminator(*shuffled_inputs)

                losses['joint'] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))

                for modality_key in self.sorted_keys:
                    discriminator = getattr(self.xz_discriminators, modality_key)[0]
                    real_x = real_inputs[modality_key]['x']
                    enc_z = gen_inputs[modality_key]['z']

                    enc_s = enc_z[:, :-self.content_dim]

                    label_value = 2
                    for other_k in self.sorted_keys:
                        if other_k == modality_key:
                            continue

                        other_enc_z = gen_inputs[other_k]['z']
                        other_enc_c = other_enc_z[:, -self.content_dim:]

                        dis_other = discriminator(real_x, torch.cat([enc_s, other_enc_c], dim=1))
                        losses['{}_c{}'.format(modality_key, label_value)] = \
                            2. / (1 + self.n_modalities) * F.cross_entropy(dis_other, label_value * label_ones)

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

                    # q(x, s, c) : p(x, s, c)
                    dis_0 = discriminator[0](real_x, enc_z)
                    dis_1 = discriminator[0](dec_x, real_z)
                    losses['{}_c0'.format(modality_key)] = \
                        2. / (1 + self.n_modalities) * F.cross_entropy(dis_0, 0 * label_ones)
                    losses['{}_c1'.format(modality_key)] = \
                        2. / (1 + self.n_modalities) * F.cross_entropy(dis_1, label_ones)

                    # shuffle x and z together
                    real_x_shuffled, enc_z_shuffled = utils.permute_dim([real_x, enc_z], dim=0)
                    enc_s_shuffled = enc_z_shuffled[:, :-self.content_dim]
                    real_z_shuffled = utils.permute_dim(real_z, dim=0)
                    real_c_shuffled = real_z_shuffled[:, -self.content_dim:]
                    cs_shuffled = torch.cat([enc_s_shuffled, real_c_shuffled], dim=1)

                    # q(x, s, c) : q(x, s) q(c)
                    dis_real = discriminator[1](real_x, enc_z)
                    dis_fake = discriminator[1](real_x_shuffled, cs_shuffled)

                    losses['{}_1'.format(modality_key)] = torch.mean(F.softplus(-dis_real)) + \
                                                          torch.mean(F.softplus(dis_fake))
        else:
            gen_inputs = {}
            with torch.set_grad_enabled(True):
                for modality_key in self.sorted_keys:
                    if self.joint_rec:
                        encoder = getattr(self.encoders, modality_key).module
                        decoder = getattr(self.decoders, modality_key)

                        real_x = real_inputs[modality_key]['x']
                        real_z = real_inputs[modality_key]['z']

                        enc_z_dist_param = encoder(real_x)
                        enc_z = utils.reparameterize(enc_z_dist_param)
                        dec_x = decoder(real_z)

                        gen_inputs[modality_key] = {'x': dec_x, 'z': enc_z, 'z_dist_param': enc_z_dist_param}
                    else:
                        encoder = getattr(self.encoders, modality_key)
                        decoder = getattr(self.decoders, modality_key)

                        real_x = real_inputs[modality_key]['x']
                        real_z = real_inputs[modality_key]['z']

                        enc_z = encoder(real_x)
                        dec_x = decoder(real_z)

                        gen_inputs[modality_key] = {'x': dec_x, 'z': enc_z}

            if joint:
                label_value = 0
                score_q = self.joint_discriminator(*[real_inputs[k]['x'] for k in self.sorted_keys])
                for modality_key in self.sorted_keys:
                    curr_inputs = {k: {} for k in self.sorted_keys}
                    content = gen_inputs[modality_key]['z'][:, -self.content_dim:]

                    for modality_key2 in self.sorted_keys:
                        style = gen_inputs[modality_key2]['z'][:, :-self.content_dim]

                        curr_inputs[modality_key2]['x'] = real_inputs[modality_key2]['x']
                        curr_inputs[modality_key2]['z'] = torch.cat([style, content], dim=1)

                    dis_score = self.calc_joint_score_jsd2(curr_inputs, score_q)
                    adv_losses = [F.cross_entropy(dis_score, i * label_ones)
                                  for i in range(dis_score.size(1)) if i != label_value]
                    losses['joint_q{}'.format(label_value)] = \
                        2. / (1 + self.n_modalities) * torch.mean(torch.stack(adv_losses, dim=0), dim=0)

                    label_value += 1

                score_p = self.joint_discriminator(*[gen_inputs[k]['x'] for k in self.sorted_keys])
                curr_inputs = {k: {} for k in self.sorted_keys}
                for modality_key in self.sorted_keys:
                    curr_inputs[modality_key]['x'] = gen_inputs[modality_key]['x']
                    curr_inputs[modality_key]['z'] = real_inputs[modality_key]['z']

                dis_score = self.calc_joint_score_jsd2(curr_inputs, score_p)
                adv_losses = [F.cross_entropy(dis_score, i * label_ones)
                              for i in range(dis_score.size(1)) if i != label_value]
                losses['joint_p'] = 2. / (1 + self.n_modalities) * torch.mean(torch.stack(adv_losses, dim=0), dim=0)

                for modality_key in self.sorted_keys:
                    discriminator = getattr(self.xz_discriminators, modality_key)[0]
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
                        adv_losses = [F.cross_entropy(dis_other, i * label_ones)
                                      for i in range(dis_other.size(1)) if i != label_value]

                        losses['{}_c{}'.format(modality_key, label_value)] = \
                            self.lambda_unimodal * \
                            2. / (1 + self.n_modalities) * torch.mean(torch.stack(adv_losses, dim=0), dim=0)

                        label_value += 1

                if self.joint_rec:
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
                            losses['{}_x_rec_joint'.format(modality_key)] = \
                                self.mod_coeff[modality_key] * self.lambda_x_rec * x_rec_loss

            else:
                for modality_key in self.sorted_keys:
                    discriminator = getattr(self.xz_discriminators, modality_key)[0]

                    real_x = real_inputs[modality_key]['x']
                    enc_z = gen_inputs[modality_key]['z']

                    dec_x = gen_inputs[modality_key]['x']
                    real_z = real_inputs[modality_key]['z']

                    # q(x, s, c)
                    dis_0 = discriminator(real_x, enc_z)
                    label_value = 0
                    adv_losses = [F.cross_entropy(dis_0, i * label_ones)
                                  for i in range(dis_0.size(1)) if i != label_value]

                    losses['{}_c0'.format(modality_key)] = \
                        self.lambda_unimodal * \
                        2. / (1 + self.n_modalities) * torch.mean(torch.stack(adv_losses, dim=0), dim=0)

                    # p(x, s, c)
                    dis_1 = discriminator(dec_x, real_z)
                    label_value = 1
                    adv_losses = [F.cross_entropy(dis_1, i * label_ones)
                                  for i in range(dis_1.size(1)) if i != label_value]
                    losses['{}_c1'.format(modality_key)] = \
                        self.lambda_unimodal * \
                        2. / (1 + self.n_modalities) * torch.mean(torch.stack(adv_losses, dim=0), dim=0)

                if not self.joint_rec:
                    if self.lambda_x_rec > 0.0:
                        for modality_key in self.sorted_keys:
                            decoder = getattr(self.decoders, modality_key)
                            x_real = real_inputs[modality_key]['x']
                            enc_z = gen_inputs[modality_key]['z']

                            x_rec = decoder(enc_z)

                            x_rec_loss = (x_rec - x_real).square().mean()

                            losses['{}_x_rec'.format(modality_key)] = \
                                self.mod_coeff[modality_key] * self.lambda_x_rec * x_rec_loss

                if (self.lambda_c_rec > 0.0 and not self.joint_rec) or self.lambda_s_rec > 0.0:
                    for modality_key in self.sorted_keys:
                        encoder = getattr(self.encoders, modality_key)
                        z_rec = encoder(gen_inputs[modality_key]['x'])

                        if self.lambda_c_rec > 0.0 and not self.joint_rec:
                            c_real = real_inputs[modality_key]['z'][:, -self.content_dim:]
                            c_rec = z_rec[:, -self.content_dim:]
                            c_rec_loss = (c_rec - c_real).square().mean()
                            losses['{}_c_rec'.format(modality_key)] = self.lambda_c_rec * c_rec_loss

                        if self.lambda_s_rec > 0.0:
                            s_real = real_inputs[modality_key]['z'][:, :-self.content_dim]
                            s_rec = z_rec[:, :-self.content_dim]
                            s_rec_loss = (s_rec - s_real).square().mean()
                            losses['{}_s_rec'.format(modality_key)] = self.lambda_s_rec * s_rec_loss

        return losses

    # def calc_score_kl(self, curr_inputs, curr_key=None, cross=False, with_cross=False):
    #     scores = {}
    #     for modality_key in self.sorted_keys:
    #         discriminator = getattr(self.xz_discriminators, modality_key)
    #         scores[modality_key] = [
    #             # q(x, s, c) : p(x, s, c)
    #             discriminator[0](curr_inputs[modality_key]['x'], curr_inputs[modality_key]['z']),
    #
    #             # q(x, s, c) : q(x, s) p(c)
    #             discriminator[1](curr_inputs[modality_key]['x'], curr_inputs[modality_key]['z'])
    #         ]
    #
    #     if curr_key is not None:
    #         # i is current key
    #
    #         if not cross:
    #             # q_i : q_j for j in keys
    #             val1 = torch.sum(torch.stack([scores[curr_key][1] - s[1] for s in scores.values()], dim=0), dim=0)
    #
    #             # q_i : p
    #             val2 = scores[curr_key][1] + torch.sum(torch.stack([s[0] - s[1]  # q(x, s) p(c) : p(x, s, c)
    #                                                                 for s in scores.values()], dim=0), dim=0)
    #             if with_cross:
    #                 pass
    #
    #             return val1 + val2
    #     else:
    #         # decoder distribution
    #         val0 = self.joint_discriminator(*[curr_inputs[k]['x'] for k in self.sorted_keys])
    #
    #         val1 = torch.sum(torch.stack([s[0] - s[1]  # q(x, s) p(c) : p(x, s, c)
    #                                       for s in scores.values()], dim=0), dim=0)
    #
    #         # val2 = -torch.sum(torch.stack([val0
    #         #                                + s[1]  # q(x, s, c) : q(x, s) p(c)
    #         #                                + val1 for s in scores.values()], dim=0), dim=0)
    #         val2 = self.n_modalities * (val0 + val1) + torch.sum(torch.stack([s[1]
    #                                                                           for s in scores.values()], dim=0), dim=0)
    #         val2 = -val2
    #         return val2
    #
    # def calc_score_kl2(self, curr_inputs, curr_key=None):
    #     scores = {}
    #     for modality_key in self.sorted_keys:
    #         discriminator = getattr(self.xz_discriminators, modality_key)
    #
    #         scores[modality_key] = [
    #             # q(x, s, c) : p(x, s, c)
    #             discriminator[0](curr_inputs[modality_key]['x'], curr_inputs[modality_key]['z']),
    #
    #             # q(x, s, c) : q(x, s) p(c)
    #             discriminator[1](curr_inputs[modality_key]['x'], curr_inputs[modality_key]['z'])
    #         ]
    #
    #     if curr_key is not None:
    #         param = curr_inputs[curr_key]['z_dist_param']
    #         param_mean = param[:, :param.size(1) // 2]
    #         param_logvar = param[:, param.size(1) // 2:]
    #         param_mean_content = param_mean[:, -self.content_dim:]
    #         param_logvar_content = param_logvar[:, -self.content_dim:]
    #
    #         sum_kld = []
    #         for label_value, other_key in enumerate(self.sorted_keys):
    #             if other_key == curr_key:
    #                 continue
    #             param2 = curr_inputs[other_key]['z_dist_param']
    #             param_mean2 = param2[:, :param.size(1) // 2]
    #             param_logvar2 = param2[:, param.size(1) // 2:]
    #             param_mean_content2 = param_mean2[:, -self.content_dim:]
    #             param_logvar_content2 = param_logvar2[:, -self.content_dim:]
    #
    #             kld = self.weights[label_value] * utils.calc_kl_divergence(param_mean_content, param_logvar_content,
    #                                                                        param_mean_content2.detach(),
    #                                                                        param_logvar_content2.detach(),
    #                                                                        dim=-1, keepdim=True)
    #             sum_kld.append(kld)
    #
    #         sum_kld = torch.sum(torch.stack(sum_kld, dim=0), dim=0)
    #
    #         kld = self.weights[-1] * utils.calc_kl_divergence(param_mean_content, param_logvar_content,
    #                                                           dim=-1, keepdim=True)
    #
    #         sum_qp = self.weights[-1] * torch.sum(torch.stack([s[0] - s[1]  # q(x, s) p(c) : p(x, s, c)
    #                                                            for s in scores.values()], dim=0), dim=0)
    #
    #         return sum_kld + kld + sum_qp
    #     else:
    #         joint = self.joint_discriminator(*[curr_inputs[k]['x'] for k in self.sorted_keys])
    #
    #         sum_qp = torch.sum(torch.stack([s[0] - s[1]  # q(x, s) p(c) : p(x, s, c)
    #                                         for s in scores.values()], dim=0), dim=0)
    #
    #         sum_qq = torch.sum(torch.stack([self.weights[i] * scores[k][1]
    #                                         for i, k in enumerate(self.sorted_keys)], dim=0), dim=0)
    #
    #         return -(joint + sum_qp + sum_qq)
    #
    # def forward_kl(self, real_inputs, train_d=True, joint=False, progress=None):
    #     self.generators.requires_grad_(not train_d)
    #     self.discriminators.requires_grad_(train_d)
    #
    #     self.weights = [0.1, 0.1, 0.8]
    #
    #     losses = {}
    #     if train_d:
    #         if joint:
    #             joint_inputs = [real_inputs[k]['x'] for k in self.sorted_keys]
    #             shuffled_inputs = [real_inputs[k]['extra_x'] for k in self.sorted_keys]
    #
    #             dis_real = self.joint_discriminator(*joint_inputs)
    #             dis_fake = self.joint_discriminator(*shuffled_inputs)
    #
    #             losses['joint'] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #         else:
    #             gen_inputs = {}
    #             with torch.set_grad_enabled(False):
    #                 for modality_key in self.sorted_keys:
    #                     encoder = getattr(self.encoders, modality_key)
    #                     decoder = getattr(self.decoders, modality_key)
    #
    #                     real_x = real_inputs[modality_key]['x']
    #                     real_z = real_inputs[modality_key]['z']
    #
    #                     enc_z = encoder(real_x)
    #                     dec_x = decoder(real_z)
    #
    #                     gen_inputs[modality_key] = {'x': dec_x, 'z': enc_z}
    #
    #             for modality_key in self.sorted_keys:
    #                 discriminator = getattr(self.xz_discriminators, modality_key)
    #
    #                 real_x = real_inputs[modality_key]['x']
    #                 enc_z = gen_inputs[modality_key]['z']
    #                 dec_x = gen_inputs[modality_key]['x']
    #                 real_z = real_inputs[modality_key]['z']
    #
    #                 # if self.lambda_mod[modality_key] * self.lambda_gp > 0.0:
    #                 #     real_x.requires_grad_(True)
    #                 #     dec_x.requires_grad_(True)
    #                 #     real_z.requires_grad_(True)
    #                 #     enc_z.requires_grad_(True)
    #
    #                 # q(x, s, c) : p(x, s, c)
    #                 dis_real = discriminator[0](real_x, enc_z)
    #                 dis_fake = discriminator[0](dec_x, real_z)
    #                 losses['{}_0'.format(modality_key)] = torch.mean(F.softplus(-dis_real)) + \
    #                                                       torch.mean(F.softplus(dis_fake))
    #
    #                 # if self.lambda_mod[modality_key] * self.lambda_gp > 0.0:
    #                 #     losses['{}_0_gp'.format(modality_key)] = utils.calc_gradient_penalty_jsd(
    #                 #         dis_real, [real_x, enc_z], dis_fake, [dec_x, real_z],
    #                 #         self.lambda_mod[modality_key] * self.lambda_gp
    #                 #     )
    #
    #                 # # shuffle x and z together
    #                 real_x_shuffled, enc_z_shuffled = utils.permute_dim([real_x, enc_z], dim=0)
    #                 enc_s_shuffled = enc_z_shuffled[:, :-self.content_dim]
    #                 real_z_shuffled = utils.permute_dim(real_z, dim=0)
    #                 real_c_shuffled = real_z_shuffled[:, -self.content_dim:]
    #                 cs_shuffled = torch.cat([enc_s_shuffled, real_c_shuffled], dim=1)
    #
    #                 # q(x, s, c) : q(x, s) p(c)
    #                 dis_real = discriminator[1](real_x, enc_z)
    #                 dis_fake = discriminator[1](real_x_shuffled, cs_shuffled)
    #                 losses['{}_1'.format(modality_key)] = torch.mean(F.softplus(-dis_real)) + \
    #                                                       torch.mean(F.softplus(dis_fake))
    #     else:
    #         gen_inputs = {}
    #         with torch.set_grad_enabled(True):
    #             for modality_key in self.sorted_keys:
    #                 encoder = getattr(self.encoders, modality_key).module
    #                 decoder = getattr(self.decoders, modality_key)
    #
    #                 real_x = real_inputs[modality_key]['x']
    #                 real_z = real_inputs[modality_key]['z']
    #
    #                 enc_z_dist_param = encoder(real_x)
    #                 enc_z = utils.reparameterize(enc_z_dist_param)
    #                 dec_x = decoder(real_z)
    #
    #                 gen_inputs[modality_key] = {'x': dec_x, 'z': enc_z, 'z_dist_param': enc_z_dist_param}
    #
    #         if joint:
    #             # encoder distributions
    #             for label_value, modality_key in enumerate(self.sorted_keys):
    #                 curr_inputs = {k: {} for k in self.sorted_keys}
    #                 content = gen_inputs[modality_key]['z'][:, -self.content_dim:]
    #
    #                 for modality_key2 in self.sorted_keys:
    #                     style = gen_inputs[modality_key2]['z'][:, :-self.content_dim]
    #
    #                     curr_inputs[modality_key2]['x'] = real_inputs[modality_key2]['x']
    #                     curr_inputs[modality_key2]['z'] = torch.cat([style, content], dim=1)
    #                     curr_inputs[modality_key2]['z_dist_param'] = gen_inputs[modality_key2]['z_dist_param']
    #
    #                 losses['joint_q{}'.format(label_value)] = self.weights[label_value] * torch.mean(
    #                     self.calc_score_kl2(curr_inputs, curr_key=modality_key))
    #
    #             # decoder distribution
    #             curr_inputs = {k: {} for k in self.sorted_keys}
    #             for modality_key in self.sorted_keys:
    #                 curr_inputs[modality_key]['x'] = gen_inputs[modality_key]['x']
    #                 curr_inputs[modality_key]['z'] = real_inputs[modality_key]['z']
    #
    #                 # encoder = getattr(self.encoders, modality_key).module
    #                 # encoder.requires_grad_(False)
    #                 # enc_z_dist_param = encoder(gen_inputs[modality_key]['x'])
    #                 # encoder.requires_grad_(True)
    #                 # curr_inputs[modality_key]['z_dist_param'] = enc_z_dist_param
    #
    #             losses['joint_p'] = self.weights[-1] * torch.mean(self.calc_score_kl2(curr_inputs))
    #         else:
    #             for modality_key in self.sorted_keys:
    #                 discriminator = getattr(self.xz_discriminators, modality_key)[0]
    #
    #                 real_x = real_inputs[modality_key]['x']
    #                 enc_z = gen_inputs[modality_key]['z']
    #
    #                 dec_x = gen_inputs[modality_key]['x']
    #                 real_z = real_inputs[modality_key]['z']
    #
    #                 dis_real = discriminator(real_x, enc_z)
    #                 dis_fake = discriminator(dec_x, real_z)
    #                 loss = torch.mean(dis_real) - torch.mean(dis_fake)
    #                 losses['{}_real_fake'.format(modality_key)] = self.lambda_unimodal * loss
    #
    #     return losses
