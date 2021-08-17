import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class FactorModel(nn.Module):
    def __init__(self, encoders, decoders, xz_discriminators, joint_discriminator,
                 content_dim=20, lambda_unimodal=0.0, lambda_x_rec=0.0, lambda_c_rec=0.0, lambda_s_rec=0.0,
                 lambda_gp=(0., 0.)):
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
        self.lambda_gp = lambda_gp

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

        score_sum = score_joint + torch.sum(torch.stack([s[:, -1] - s[:, 1]  # q(x, s) p(c)/ p(x, s, c)
                                                         for s in scores.values()], dim=0), dim=0)

        joint_score = []
        for modality_key in self.sorted_keys:
            s = scores[modality_key]
            # q(x, s, c) / (q(x, s) p(c))
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
                with torch.set_grad_enabled(True):
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
            with torch.set_grad_enabled(not train_d):
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
                    losses['joint_q{}'.format(label_value)] = torch.mean(torch.stack(adv_losses, dim=0), dim=0)

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
                losses['joint_p{}'.format(label_value)] = torch.mean(torch.stack(adv_losses, dim=0), dim=0)
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

                        losses['{}_c{}'.format(modality_key, label_value)] = self.lambda_unimodal * torch.mean(
                            torch.stack(adv_losses, dim=0),
                            dim=0)
                        label_value += 1

                if self.lambda_c_rec > 0.0:
                    z_rec_dist_params = []

                    for k in self.sorted_keys:
                        encoder = getattr(self.encoders, k).module
                        z_rec_dist_param = encoder(gen_inputs[k]['x'])
                        z_rec_dist_params.append(z_rec_dist_param)

                    z_joint_dist_param = utils.joint_posterior(*z_rec_dist_params, average=True)
                    z_joint_dist_param_mean = z_joint_dist_param[:, :z_joint_dist_param.size(1) // 2]
                    c_joint_dist_param_mean = z_joint_dist_param_mean[:, -self.content_dim:]

                    c_real = list(real_inputs.values())[0]['z'][:, -self.content_dim:]

                    c_rec_loss = (c_joint_dist_param_mean - c_real).square().mean()
                    losses['joint_c_rec'] = self.lambda_c_rec * c_rec_loss

                if self.lambda_x_rec > 0.0:
                    z_joint = utils.reparameterize(
                        utils.joint_posterior(*[v['z_dist_param'] for v in gen_inputs.values()], average=True))

                    c_joint = z_joint[:, -self.content_dim:]

                    for k in self.sorted_keys:
                        decoder = getattr(self.decoders, k)

                        x_real = real_inputs[k]['x']
                        x_rec = decoder(torch.cat([gen_inputs[k]['z'][:, :-self.content_dim], c_joint], dim=1))

                        x_rec_loss = (x_rec - x_real).square().mean()
                        losses['{}_x_rec_joint'.format(k)] = self.lambda_x_rec * x_rec_loss
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

                    losses['{}_c0'.format(modality_key)] = self.lambda_unimodal * torch.mean(torch.stack(adv_losses,
                                                                                                         dim=0), dim=0)

                    # p(x, s, c)
                    dis_1 = discriminator(dec_x, real_z)
                    dis_1 = dis_1[:, :-1]
                    label_value = 1
                    adv_losses = [F.cross_entropy(dis_1, i * label_ones)
                                  for i in range(dis_1.size(1)) if i != label_value]
                    losses['{}_c1'.format(modality_key)] = self.lambda_unimodal * torch.mean(torch.stack(adv_losses,
                                                                                                         dim=0), dim=0)

                if self.lambda_x_rec > 0.0:
                    for k in self.sorted_keys:
                        decoder = getattr(self.decoders, k)
                        x_real = real_inputs[k]['x']
                        z_enc = gen_inputs[k]['z']
                        x_rec = decoder(z_enc)

                        x_rec_loss = (x_rec - x_real).square().mean()
                        losses['{}_x_rec_unimodal'.format(k)] = self.lambda_x_rec * x_rec_loss

                if self.lambda_s_rec > 0.0:
                    for k in self.sorted_keys:
                        encoder = getattr(self.encoders, k).module
                        z_rec_dist_param = encoder(gen_inputs[k]['x'])
                        z_rec_dist_param_mean = z_rec_dist_param[:, :z_rec_dist_param.size(1) // 2]
                        s_rec_dist_param_mean = z_rec_dist_param_mean[:, :-self.content_dim]

                        s_real = real_inputs[k]['z'][:, :-self.content_dim]

                        s_rec_loss = (s_rec_dist_param_mean - s_real).square().mean()
                        losses['{}_s_rec'.format(k)] = self.lambda_s_rec * s_rec_loss

        return losses

    # def forward_kl(self, real_inputs, train_d=True, joint=False, progress=None):
    #     self.generators.requires_grad_(not train_d)
    #     self.discriminators.requires_grad_(train_d)
    #
    #     batch_size = list(real_inputs.values())[0]['x'].size(0)
    #     device = list(real_inputs.values())[0]['x'].device
    #     label_zeros = torch.zeros(batch_size, dtype=torch.long, device=device)
    #     label_ones = torch.ones(batch_size, dtype=torch.long, device=device)
    #
    #     gen_inputs = {}
    #     with torch.set_grad_enabled(not train_d):
    #         for modality_key in self.sorted_keys:
    #             encoder = getattr(self.encoders, modality_key)
    #             decoder = getattr(self.decoders, modality_key)
    #
    #             real_x = real_inputs[modality_key]['x']
    #             real_z = real_inputs[modality_key]['z']
    #
    #             enc_z = encoder(real_x)
    #             dec_x = decoder(real_z)
    #
    #             gen_inputs[modality_key] = {'x': dec_x, 'z': enc_z}
    #     losses = {}
    #     if train_d:
    #         if joint:
    #             # make sure the order of sorted keys matches the input order of joint discriminator
    #             joint_inputs = [real_inputs[k]['x'] for k in self.sorted_keys]
    #             shuffled_inputs = [real_inputs[k]['extra_x'] for k in self.sorted_keys]
    #
    #             dis_real = self.joint_discriminator(*joint_inputs)
    #             dis_fake = self.joint_discriminator(*shuffled_inputs)
    #             losses['joint'] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #
    #             # for modality_key in self.sorted_keys:
    #             #     discriminator = getattr(self.xz_discriminators, modality_key)
    #             #     real_x = real_inputs[modality_key]['x']
    #             #     enc_z = gen_inputs[modality_key]['z']
    #             #
    #             #     enc_s = enc_z[:, :-self.content_dim]
    #             #
    #             #     # for other keys
    #             #     label_value = 2
    #             #     for other_k in self.sorted_keys:
    #             #         # q_j(x_i, s_i, c) where j != i
    #             #         if other_k == modality_key:
    #             #             continue
    #             #
    #             #         other_enc_z = gen_inputs[other_k]['z']
    #             #         other_enc_c = other_enc_z[:, -self.content_dim:]
    #             #         z_combined = torch.cat([enc_s, other_enc_c], dim=1)
    #             #
    #             #         # q(x, s, other_c)
    #             #         dis_other = discriminator(real_x, z_combined)
    #             #         losses['{}_c{}'.format(modality_key, label_value)] = F.cross_entropy(dis_other,
    #             #                                                                              label_value * label_ones)
    #             #         label_value += 1
    #         else:
    #             for modality_key in self.sorted_keys:
    #                 discriminator = getattr(self.xz_discriminators, modality_key)
    #
    #                 real_x = real_inputs[modality_key]['x']
    #                 enc_z = gen_inputs[modality_key]['z']
    #
    #                 dec_x = gen_inputs[modality_key]['x']
    #                 real_z = real_inputs[modality_key]['z']
    #
    #                 # shuffle x and z together
    #                 real_x_shuffled, enc_z_shuffled = utils.permute_dim([real_x, enc_z], dim=0)
    #                 enc_s_shuffled = enc_z_shuffled[:, :-self.content_dim]
    #
    #                 real_z_shuffled = utils.permute_dim(real_z, dim=0)
    #                 real_c_shuffled = real_z_shuffled[:, -self.content_dim:]
    #
    #                 cs_shuffled = torch.cat([enc_s_shuffled, real_c_shuffled], dim=1)
    #
    #                 # q(x, s, c)
    #                 losses['{}_c0'.format(modality_key)] = F.cross_entropy(discriminator(real_x, enc_z), label_zeros)
    #
    #                 # p(x, s, c)
    #                 losses['{}_c1'.format(modality_key)] = F.cross_entropy(discriminator(dec_x, real_z), label_ones)
    #
    #                 # q(x, s) p(c)
    #                 losses['{}_c{}'.format(modality_key, 2)] = F.cross_entropy(
    #                     discriminator(real_x_shuffled, cs_shuffled), 2 * label_ones)
    #
    #     else:
    #         if joint:
    #             # encoder distributions
    #             for modality_key in self.sorted_keys:
    #                 # prepare inputs
    #                 curr_inputs = {k: {} for k in self.sorted_keys}
    #
    #                 # use k2's style code + k's content code
    #                 enc_c = gen_inputs[modality_key]['z'][:, -self.content_dim:]
    #                 for modality_key2 in self.sorted_keys:
    #                     enc_s = gen_inputs[modality_key2]['z'][:, :-self.content_dim]
    #
    #                     curr_inputs[modality_key2]['x'] = real_inputs[modality_key2]['x']
    #                     curr_inputs[modality_key2]['z'] = torch.cat([enc_s, enc_c], dim=1)
    #
    #                 # calculate scores
    #                 scores = {}
    #                 for modality_key2 in self.sorted_keys:
    #                     discriminator = getattr(self.xz_discriminators, modality_key2)
    #                     scores[modality_key2] = discriminator(curr_inputs[modality_key2]['x'],
    #                                                           curr_inputs[modality_key2]['z'])
    #
    #                 # val_1 = (self.n_modalities + 1) * torch.mean(scores[modality_key][:, 0]
    #                 #                                              - scores[modality_key][:, -1])
    #                 #
    #                 # self._debug_values['{}_val_1'.format(modality_key)] = val_1
    #                 #
    #                 # val_2 = [torch.mean(scores[k][:, -1] - scores[k][:, 0]) for k in self.sorted_keys]
    #                 # for i, val in enumerate(val_2):
    #                 #     self._debug_values['{}_val_2_{}'.format(modality_key, i)] = val
    #                 # val_2 = torch.sum(torch.stack(val_2, dim=0), dim=0)
    #                 #
    #                 # val_3 = [torch.mean(scores[k][:, -1] - scores[k][:, 1]) for k in self.sorted_keys]
    #                 # for i, val in enumerate(val_3):
    #                 #     self._debug_values['{}_val_3_{}'.format(modality_key, i)] = val
    #                 # val_3 = torch.sum(torch.stack(val_3, dim=0), dim=0)
    #
    #                 losses['joint_q_{}'.format(modality_key)] = (
    #                         (self.n_modalities + 1) * torch.mean(scores[modality_key][:, 0]  # q(x, s, c) / q(x, s) p(c)
    #                                                              - scores[modality_key][:, -1])
    #                         # q(x, s) p(c) / q(x, s, c)
    #                         + torch.sum(torch.stack([torch.mean(scores[k][:, -1] - scores[k][:, 0])
    #                                                  for k in self.sorted_keys], dim=0), dim=0)
    #                         # q(x, s) p(c) / p(x, s, c)
    #                         + torch.sum(torch.stack([torch.mean(scores[k][:, -1] - scores[k][:, 1])
    #                                                  for k in self.sorted_keys], dim=0), dim=0)
    #                     #     val_1 + val_2 + val_3
    #                 )
    #
    #             # decoder distribution
    #             # prepare inputs
    #             curr_inputs = {k: {} for k in self.sorted_keys}
    #             for modality_key in self.sorted_keys:
    #                 curr_inputs[modality_key]['x'] = gen_inputs[modality_key]['x']
    #                 curr_inputs[modality_key]['z'] = real_inputs[modality_key]['z']
    #
    #             # calculate scores
    #             score_joint = -self.joint_discriminator(*[curr_inputs[k]['x'] for k in self.sorted_keys])
    #
    #             scores = {}
    #             for modality_key in self.sorted_keys:
    #                 discriminator = getattr(self.xz_discriminators, modality_key)
    #                 scores[modality_key] = discriminator(curr_inputs[modality_key]['x'],
    #                                                      curr_inputs[modality_key]['z'])
    #
    #             # val_4 = self.n_modalities * torch.mean(score_joint)
    #             # self._debug_values['val_4'.format(modality_key)] = val_4
    #             #
    #             # val_5 = [torch.mean(scores[k][:, -1] - scores[k][:, 0]) for k in self.sorted_keys]
    #             # for i, val in enumerate(val_5):
    #             #     self._debug_values['val_5_{}'.format(i)] = val
    #             #
    #             # val_5 = torch.sum(torch.stack(val_5, dim=0), dim=0)
    #             #
    #             # val_6 = [torch.mean(scores[k][:, 1] - scores[k][:, -1]) for k in self.sorted_keys]
    #             # for i, val in enumerate(val_6):
    #             #     self._debug_values['val_6_{}'.format(i)] = val
    #             #
    #             # val_6 = self.n_modalities * torch.sum(torch.stack(val_6, dim=0), dim=0)
    #
    #             losses['joint_p'] = torch.mean(
    #                 self.n_modalities * torch.mean(score_joint)
    #                 # q(x, s) p(c) / q(x, s, c)
    #                 + torch.sum(torch.stack([torch.mean(scores[k][:, -1] - scores[k][:, 0])
    #                                          for k in self.sorted_keys], dim=0), dim=0)
    #
    #                 # p(x, s, c) / q(x, s) p(c)
    #                 + self.n_modalities * torch.sum(torch.stack([torch.mean(scores[k][:, 1] - scores[k][:, -1])
    #                                                              for k in self.sorted_keys], dim=0), dim=0)
    #                 # val_4 + val_5 + val_6
    #             )
    #
    #             # for modality_key in self.sorted_keys:
    #             #     discriminator = getattr(self.xz_discriminators, modality_key)
    #             #     real_x = real_inputs[modality_key]['x']
    #             #     enc_z = gen_inputs[modality_key]['z']
    #             #     enc_s = enc_z[:, :-self.content_dim]
    #             #
    #             #     # for other keys
    #             #     label_value = 2
    #             #     for other_k in self.sorted_keys:
    #             #         # q_j(x_i, s_i, c) where j != i
    #             #         if other_k == modality_key:
    #             #             continue
    #             #
    #             #         other_enc_z = gen_inputs[other_k]['z']
    #             #         other_enc_c = other_enc_z[:, -self.content_dim:]
    #             #
    #             #         dis_other = discriminator(real_x, torch.cat([enc_s, other_enc_c], dim=1))
    #             #         # dis_other = dis_other[:, :-1]
    #             #         losses['{}_c{}'.format(modality_key, label_value)] = torch.sum(torch.stack(
    #             #             [torch.mean(dis_other[:, label_value] - dis_other[:, other_label])
    #             #              for other_label in range(dis_other.size(1)) if other_label != label_value], dim=0), dim=0)
    #             #
    #             #         label_value += 1
    #
    #             # if self.lambda_c_rec > 0.0:
    #             #     z_rec_dist_params = []
    #             #
    #             #     for k in real_inputs.keys():
    #             #         encoder = getattr(self.encoders, k).module
    #             #         z_rec_dist_param = encoder(gen_inputs[k]['x'])
    #             #         z_rec_dist_params.append(z_rec_dist_param)
    #             #
    #             #     z_joint_dist_param = utils.joint_posterior(*z_rec_dist_params, average=True)
    #             #     z_joint_dist_param_mean = z_joint_dist_param[:, :z_joint_dist_param.size(1) // 2]
    #             #     c_joint_dist_param_mean = z_joint_dist_param_mean[:, -self.content_dim:]
    #             #
    #             #     c_real = list(real_inputs.values())[0]['z'][:, -self.content_dim:]
    #             #
    #             #     c_rec_loss = (c_joint_dist_param_mean - c_real).square().mean()
    #             #     losses['joint_c_rec'] = self.lambda_c_rec * c_rec_loss
    #             #
    #             # if self.lambda_x_rec > 0.0:
    #             #     z_joint = utils.reparameterize(
    #             #         utils.joint_posterior(*[v['z_dist_param'] for v in gen_inputs.values()], average=True))
    #             #
    #             #     c_joint = z_joint[:, -self.content_dim:]
    #             #
    #             #     for k in real_inputs.keys():
    #             #         decoder = getattr(self.decoders, k)
    #             #
    #             #         x_real = real_inputs[k]['x']
    #             #         x_rec = decoder(torch.cat([gen_inputs[k]['z'][:, :-self.content_dim], c_joint], dim=1))
    #             #
    #             #         x_rec_loss = (x_rec - x_real).square().mean()
    #             #         losses['{}_x_rec'.format(k)] = self.lambda_x_rec * x_rec_loss
    #         else:
    #             for modality_key in self.sorted_keys:
    #                 discriminator = getattr(self.xz_discriminators, modality_key)
    #
    #                 real_x = real_inputs[modality_key]['x']
    #                 enc_z = gen_inputs[modality_key]['z']
    #
    #                 dec_x = gen_inputs[modality_key]['x']
    #                 real_z = real_inputs[modality_key]['z']
    #
    #                 # q(x, s, c)
    #                 dis_0 = discriminator(real_x, enc_z)
    #                 # dis_0 = dis_0[:, :-1]
    #                 losses['{}_c0'.format(modality_key)] = torch.sum(torch.stack(
    #                     [torch.mean(dis_0[:, 0] - dis_0[:, other_label]) for other_label in range(dis_0.size(1))
    #                      if other_label != 0], dim=0), dim=0)
    #
    #                 # p(x, s, c)
    #                 dis_1 = discriminator(dec_x, real_z)
    #                 # dis_1 = dis_1[:, :-1]
    #                 losses['{}_c1'.format(modality_key)] = torch.sum(torch.stack(
    #                     [torch.mean(dis_1[:, 1] - dis_1[:, other_label]) for other_label in range(dis_1.size(1))
    #                      if other_label != 1], dim=0), dim=0)
    #
    #             # if self.lambda_s_rec > 0.0:
    #             #     for k in real_inputs.keys():
    #             #         encoder = getattr(self.encoders, k).module
    #             #         z_rec_dist_param = encoder(gen_inputs[k]['x'])
    #             #         z_rec_dist_param_mean = z_rec_dist_param[:, :z_rec_dist_param.size(1) // 2]
    #             #         s_rec_dist_param_mean = z_rec_dist_param_mean[:, :-self.content_dim]
    #             #
    #             #         s_real = real_inputs[k]['z'][:, :-self.content_dim]
    #             #
    #             #         s_rec_loss = (s_rec_dist_param_mean - s_real).square().mean()
    #             #         losses['{}_s_rec'.format(k)] = self.lambda_s_rec * s_rec_loss
    #
    #     return losses
    #
    # def forward_kl2(self, real_inputs, train_d=True, joint=False, progress=None):
    #     progress = 1.
    #
    #     self.generators.requires_grad_(not train_d)
    #     self.discriminators.requires_grad_(train_d)
    #
    #     batch_size = list(real_inputs.values())[0]['x'].size(0)
    #     device = list(real_inputs.values())[0]['x'].device
    #     label_zeros = torch.zeros(batch_size, dtype=torch.long, device=device)
    #     label_ones = torch.ones(batch_size, dtype=torch.long, device=device)
    #
    #     gen_inputs = {}
    #     with torch.set_grad_enabled(not train_d):
    #         for modality_key in self.sorted_keys:
    #             encoder = getattr(self.encoders, modality_key)
    #             decoder = getattr(self.decoders, modality_key)
    #
    #             real_x = real_inputs[modality_key]['x']
    #             real_z = real_inputs[modality_key]['z']
    #
    #             enc_z = encoder(real_x)
    #             dec_x = decoder(real_z)
    #
    #             gen_inputs[modality_key] = {'x': dec_x, 'z': enc_z}
    #     losses = {}
    #     if train_d:
    #         if joint:
    #             # make sure the order of sorted keys matches the input order of joint discriminator
    #             joint_inputs = [real_inputs[k]['x'] + torch.randn_like(real_inputs[k]['x']) * 0.1 * (1 - progress)
    #                             for k in self.sorted_keys]
    #             shuffled_inputs = [
    #                 real_inputs[k]['extra_x'] + torch.randn_like(real_inputs[k]['extra_x']) * 0.1 * (1 - progress)
    #                 for k in self.sorted_keys]
    #
    #             dis_real = self.joint_discriminator(*joint_inputs)
    #             dis_fake = self.joint_discriminator(*shuffled_inputs)
    #             losses['joint'] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #
    #             # for modality_key in self.sorted_keys:
    #             #     discriminator = getattr(self.xz_discriminators, modality_key)
    #             #     real_x = real_inputs[modality_key]['x']
    #             #     enc_z = gen_inputs[modality_key]['z']
    #             #
    #             #     enc_s = enc_z[:, :-self.content_dim]
    #             #
    #             #     # for other keys
    #             #     label_value = 2
    #             #     for other_k in self.sorted_keys:
    #             #         # q_j(x_i, s_i, c) where j != i
    #             #         if other_k == modality_key:
    #             #             continue
    #             #
    #             #         other_enc_z = gen_inputs[other_k]['z']
    #             #         other_enc_c = other_enc_z[:, -self.content_dim:]
    #             #         z_combined = torch.cat([enc_s, other_enc_c], dim=1)
    #             #
    #             #         noise_x = torch.randn_like(real_x) * 0.1 * (1. - progress)
    #             #         noise_z = torch.randn_like(z_combined) * 0.1 * (1. - progress)
    #             #         # q(x, s, other_c)
    #             #         dis_other = discriminator(real_x + noise_x, z_combined + noise_z)
    #             #         losses['{}_c{}'.format(modality_key, label_value)] = F.cross_entropy(dis_other,
    #             #                                                                              label_value * label_ones)
    #             #         dis_val = torch.mean(dis_other, dim=0, keepdim=True)
    #             #         for i in range(dis_val.size(1)):
    #             #             self._debug_values['{}_dis_2_{}'.format(modality_key, i)] = dis_val[0, i]
    #             #         label_value += 1
    #         else:
    #             for modality_key in self.sorted_keys:
    #                 discriminator = getattr(self.xz_discriminators, modality_key)
    #
    #                 real_x = real_inputs[modality_key]['x']
    #                 enc_z = gen_inputs[modality_key]['z']
    #
    #                 dec_x = gen_inputs[modality_key]['x']
    #                 real_z = real_inputs[modality_key]['z']
    #
    #                 # shuffle x and z together
    #                 real_x_shuffled, enc_z_shuffled = utils.permute_dim([real_x, enc_z], dim=0)
    #                 enc_s_shuffled = enc_z_shuffled[:, :-self.content_dim]
    #
    #                 real_z_shuffled = utils.permute_dim(real_z, dim=0)
    #                 real_c_shuffled = real_z_shuffled[:, -self.content_dim:]
    #
    #                 cs_shuffled = torch.cat([enc_s_shuffled, real_c_shuffled], dim=1)
    #
    #                 noise_x = torch.randn_like(real_x) * 0.1 * (1. - progress)
    #                 noise_z = torch.randn_like(enc_z) * 0.1 * (1. - progress)
    #                 dis_val = discriminator(real_x + noise_x, enc_z + noise_z)
    #                 # q(x, s, c)
    #                 losses['{}_c0'.format(modality_key)] = F.cross_entropy(dis_val, label_zeros)
    #                 dis_val = torch.mean(dis_val, dim=0, keepdim=True)
    #                 for i in range(dis_val.size(1)):
    #                     self._debug_values['{}_dis_0_{}'.format(modality_key, i)] = dis_val[0, i]
    #
    #                 noise_x = torch.randn_like(dec_x) * 0.1 * (1. - progress)
    #                 noise_z = torch.randn_like(real_z) * 0.1 * (1. - progress)
    #                 dis_val = discriminator(dec_x + noise_x, real_z + noise_z)
    #                 # p(x, s, c)
    #                 losses['{}_c1'.format(modality_key)] = F.cross_entropy(discriminator(dec_x, real_z), label_ones)
    #                 dis_val = torch.mean(dis_val, dim=0, keepdim=True)
    #                 for i in range(dis_val.size(1)):
    #                     self._debug_values['{}_dis_1_{}'.format(modality_key, i)] = dis_val[0, i]
    #
    #                 noise_x = torch.randn_like(real_x_shuffled) * 0.1 * (1. - progress)
    #                 noise_z = torch.randn_like(cs_shuffled) * 0.1 * (1. - progress)
    #                 dis_val = discriminator(real_x_shuffled + noise_x, cs_shuffled + noise_z)
    #                 # q(x, s) p(c)
    #                 losses['{}_c{}'.format(modality_key, self.n_modalities)] = F.cross_entropy(
    #                     dis_val, self.n_modalities * label_ones)
    #                 for i in range(dis_val.size(1)):
    #                     self._debug_values['{}_dis_3_{}'.format(modality_key, i)] = dis_val[0, i]
    #
    #                 self._debug_values['{}_enc_z'.format(modality_key)] = torch.mean(enc_z)
    #                 self._debug_values['{}_dec_x'.format(modality_key)] = torch.mean(dec_x)
    #     else:
    #         if joint:
    #             # encoder distributions
    #             for modality_key in self.sorted_keys:
    #                 # prepare inputs
    #                 curr_inputs = {k: {} for k in self.sorted_keys}
    #
    #                 # use k2's style code + k's content code
    #                 enc_c = gen_inputs[modality_key]['z'][:, -self.content_dim:]
    #                 for modality_key2 in self.sorted_keys:
    #                     enc_s = gen_inputs[modality_key2]['z'][:, :-self.content_dim]
    #
    #                     curr_inputs[modality_key2]['x'] = real_inputs[modality_key2]['x']
    #                     curr_inputs[modality_key2]['z'] = torch.cat([enc_s, enc_c], dim=1)
    #
    #                 # calculate scores
    #                 scores = {}
    #                 for modality_key2 in self.sorted_keys:
    #                     discriminator = getattr(self.xz_discriminators, modality_key2)
    #                     noise_x = torch.randn_like(curr_inputs[modality_key2]['x']) * 0.1 * (1. - progress)
    #                     noise_z = torch.randn_like(curr_inputs[modality_key2]['z']) * 0.1 * (1. - progress)
    #                     scores[modality_key2] = discriminator(curr_inputs[modality_key2]['x'] + noise_x,
    #                                                           curr_inputs[modality_key2]['z'] + noise_z)
    #
    #                 val_1 = (self.n_modalities + 1) * torch.mean(scores[modality_key][:, 0]
    #                                                              - scores[modality_key][:, -1])
    #
    #                 self._debug_values['{}_val_1'.format(modality_key)] = val_1
    #
    #                 val_2 = [torch.mean(scores[k][:, -1] - scores[k][:, 0]) for k in self.sorted_keys]
    #                 for i, val in enumerate(val_2):
    #                     self._debug_values['{}_val_2_{}'.format(modality_key, i)] = val
    #                 val_2 = torch.sum(torch.stack(val_2, dim=0), dim=0)
    #
    #                 val_3 = [torch.mean(scores[k][:, -1] - scores[k][:, 1]) for k in self.sorted_keys]
    #                 for i, val in enumerate(val_3):
    #                     self._debug_values['{}_val_3_{}'.format(modality_key, i)] = val
    #                 val_3 = torch.sum(torch.stack(val_3, dim=0), dim=0)
    #
    #                 losses['joint_q_{}'.format(modality_key)] = (
    #                     # (self.n_modalities + 1) * torch.mean(scores[modality_key][:, 0]  # q(x, s, c) / q(x, s) p(c)
    #                     #                                      - scores[modality_key][:, -1])
    #                     # # q(x, s) p(c) / q(x, s, c)
    #                     # + torch.sum(torch.stack([torch.mean(scores[k][:, -1] - scores[k][:, 0])
    #                     #                          for k in self.sorted_keys], dim=0), dim=0)
    #                     # # q(x, s) p(c) / p(x, s, c)
    #                     # + torch.sum(torch.stack([torch.mean(scores[k][:, -1] - scores[k][:, 1])
    #                     #                          for k in self.sorted_keys], dim=0), dim=0)
    #                         val_1 + val_2 + val_3
    #                 )
    #
    #             # decoder distribution
    #             # prepare inputs
    #             curr_inputs = {k: {} for k in self.sorted_keys}
    #             for modality_key in self.sorted_keys:
    #                 curr_inputs[modality_key]['x'] = gen_inputs[modality_key]['x']
    #                 curr_inputs[modality_key]['z'] = real_inputs[modality_key]['z']
    #
    #             # calculate scores
    #             score_joint = -self.joint_discriminator(
    #                 *[curr_inputs[k]['x'] + torch.randn_like(curr_inputs[k]['x']) * 0.1 * (1 - progress) for k in
    #                   self.sorted_keys])
    #
    #             scores = {}
    #             for modality_key in self.sorted_keys:
    #                 discriminator = getattr(self.xz_discriminators, modality_key)
    #                 noise_x = torch.randn_like(curr_inputs[modality_key]['x']) * 0.1 * (1. - progress)
    #                 noise_z = torch.randn_like(curr_inputs[modality_key]['z']) * 0.1 * (1. - progress)
    #                 scores[modality_key] = discriminator(curr_inputs[modality_key]['x'] + noise_x,
    #                                                      curr_inputs[modality_key]['z'] + noise_z)
    #
    #             val_4 = self.n_modalities * torch.mean(score_joint)
    #             self._debug_values['val_4'.format(modality_key)] = val_4
    #
    #             val_5 = [torch.mean(scores[k][:, -1] - scores[k][:, 0]) for k in self.sorted_keys]
    #             for i, val in enumerate(val_5):
    #                 self._debug_values['val_5_{}'.format(i)] = val
    #
    #             val_5 = torch.sum(torch.stack(val_5, dim=0), dim=0)
    #
    #             val_6 = [torch.mean(scores[k][:, 1] - scores[k][:, -1]) for k in self.sorted_keys]
    #             for i, val in enumerate(val_6):
    #                 self._debug_values['val_6_{}'.format(i)] = val
    #
    #             val_6 = self.n_modalities * torch.sum(torch.stack(val_6, dim=0), dim=0)
    #
    #             losses['joint_p'] = torch.mean(
    #                 # self.n_modalities * torch.mean(score_joint)
    #                 # # q(x, s) p(c) / q(x, s, c)
    #                 # + torch.sum(torch.stack([torch.mean(scores[k][:, -1] - scores[k][:, 0])
    #                 #                          for k in self.sorted_keys], dim=0), dim=0)
    #                 #
    #                 # # p(x, s, c) / q(x, s) p(c)
    #                 # + self.n_modalities * torch.sum(torch.stack([torch.mean(scores[k][:, 1] - scores[k][:, -1])
    #                 #                                              for k in self.sorted_keys], dim=0), dim=0)
    #                 val_4 + val_5 + val_6
    #             )
    #
    #             # for modality_key in self.sorted_keys:
    #             #     discriminator = getattr(self.xz_discriminators, modality_key)
    #             #     real_x = real_inputs[modality_key]['x']
    #             #     enc_z = gen_inputs[modality_key]['z']
    #             #     enc_s = enc_z[:, :-self.content_dim]
    #             #
    #             #     # for other keys
    #             #     label_value = 2
    #             #     for other_k in self.sorted_keys:
    #             #         # q_j(x_i, s_i, c) where j != i
    #             #         if other_k == modality_key:
    #             #             continue
    #             #
    #             #         other_enc_z = gen_inputs[other_k]['z']
    #             #         other_enc_c = other_enc_z[:, -self.content_dim:]
    #             #
    #             #         dis_other = discriminator(real_x, torch.cat([enc_s, other_enc_c], dim=1))
    #             #         dis_other = dis_other[:, :-1]
    #             #         losses['{}_c{}'.format(modality_key, label_value)] = torch.sum(torch.stack(
    #             #             [torch.mean(dis_other[:, label_value] - dis_other[:, other_label])
    #             #              for other_label in range(dis_other.size(1)) if other_label != label_value], dim=0), dim=0)
    #             #
    #             #         label_value += 1
    #
    #             # if self.lambda_c_rec > 0.0:
    #             #     z_rec_dist_params = []
    #             #
    #             #     for k in real_inputs.keys():
    #             #         encoder = getattr(self.encoders, k).module
    #             #         z_rec_dist_param = encoder(gen_inputs[k]['x'])
    #             #         z_rec_dist_params.append(z_rec_dist_param)
    #             #
    #             #     z_joint_dist_param = utils.joint_posterior(*z_rec_dist_params, average=True)
    #             #     z_joint_dist_param_mean = z_joint_dist_param[:, :z_joint_dist_param.size(1) // 2]
    #             #     c_joint_dist_param_mean = z_joint_dist_param_mean[:, -self.content_dim:]
    #             #
    #             #     c_real = list(real_inputs.values())[0]['z'][:, -self.content_dim:]
    #             #
    #             #     c_rec_loss = (c_joint_dist_param_mean - c_real).square().mean()
    #             #     losses['joint_c_rec'] = self.lambda_c_rec * c_rec_loss
    #             #
    #             # if self.lambda_x_rec > 0.0:
    #             #     z_joint = utils.reparameterize(
    #             #         utils.joint_posterior(*[v['z_dist_param'] for v in gen_inputs.values()], average=True))
    #             #
    #             #     c_joint = z_joint[:, -self.content_dim:]
    #             #
    #             #     for k in real_inputs.keys():
    #             #         decoder = getattr(self.decoders, k)
    #             #
    #             #         x_real = real_inputs[k]['x']
    #             #         x_rec = decoder(torch.cat([gen_inputs[k]['z'][:, :-self.content_dim], c_joint], dim=1))
    #             #
    #             #         x_rec_loss = (x_rec - x_real).square().mean()
    #             #         losses['{}_x_rec'.format(k)] = self.lambda_x_rec * x_rec_loss
    #         else:
    #             for modality_key in self.sorted_keys:
    #                 discriminator = getattr(self.xz_discriminators, modality_key)
    #
    #                 real_x = real_inputs[modality_key]['x']
    #                 enc_z = gen_inputs[modality_key]['z']
    #
    #                 dec_x = gen_inputs[modality_key]['x']
    #                 real_z = real_inputs[modality_key]['z']
    #
    #                 # q(x, s, c)
    #                 dis_0 = discriminator(real_x, enc_z)
    #                 dis_0 = dis_0[:, :-1]
    #                 losses['{}_c0'.format(modality_key)] = torch.sum(torch.stack(
    #                     [torch.mean(dis_0[:, 0] - dis_0[:, other_label]) for other_label in range(dis_0.size(1))
    #                      if other_label != 0], dim=0), dim=0)
    #
    #                 # p(x, s, c)
    #                 dis_1 = discriminator(dec_x, real_z)
    #                 dis_1 = dis_1[:, :-1]
    #                 losses['{}_c1'.format(modality_key)] = torch.sum(torch.stack(
    #                     [torch.mean(dis_1[:, 1] - dis_1[:, other_label]) for other_label in range(dis_1.size(1))
    #                      if other_label != 1], dim=0), dim=0)
    #
    #             # if self.lambda_s_rec > 0.0:
    #             #     for k in real_inputs.keys():
    #             #         encoder = getattr(self.encoders, k).module
    #             #         z_rec_dist_param = encoder(gen_inputs[k]['x'])
    #             #         z_rec_dist_param_mean = z_rec_dist_param[:, :z_rec_dist_param.size(1) // 2]
    #             #         s_rec_dist_param_mean = z_rec_dist_param_mean[:, :-self.content_dim]
    #             #
    #             #         s_real = real_inputs[k]['z'][:, :-self.content_dim]
    #             #
    #             #         s_rec_loss = (s_rec_dist_param_mean - s_real).square().mean()
    #             #         losses['{}_s_rec'.format(k)] = self.lambda_s_rec * s_rec_loss
    #
    #     return losses

    # def forward3(self, real_inputs, train_d=True, joint=False, progress=None):
    #     self.generators.requires_grad_(not train_d)
    #     self.discriminators.requires_grad_(train_d)
    #
    #     # real_inputs: key -> [real_x, real_z] or key -> [paired_x, unpaired_x]
    #     batch_size = list(real_inputs.values())[0][0].size(0)
    #     device = list(real_inputs.values())[0][0].device
    #     label_ones = torch.ones(batch_size, dtype=torch.long, device=device)
    #
    #     if not (train_d and joint):
    #         with torch.set_grad_enabled(not train_d):
    #             # inputs: key -> [x, z] where z = style + content
    #             gen_inputs = {}
    #             for k, v in real_inputs.items():
    #                 encoder = getattr(self.encoders, k).module
    #                 decoder = getattr(self.decoders, k)
    #                 enc_z_dist_param = encoder(v[0])
    #                 enc_z = utils.reparameterize(enc_z_dist_param)
    #                 dec_x = decoder(v[1])
    #                 gen_inputs[k] = [dec_x, enc_z, enc_z_dist_param]
    #
    #     losses = {}
    #     if train_d:
    #         if joint:
    #             # all modalities should be available
    #             assert len(real_inputs.items()) == self.n_modalities
    #
    #             # make sure the order of sorted keys matches the input order of joint discriminator
    #             joint_inputs = [real_inputs[k][0] for k in sorted(real_inputs.keys())]
    #             shuffled_inputs = [utils.permute_dim(real_inputs[k][1], dim=0) for k in sorted(real_inputs.keys())]
    #
    #             dis_real = self.joint_discriminator(*joint_inputs)
    #             dis_fake = self.joint_discriminator(*shuffled_inputs)
    #
    #             losses['joint'] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #         else:
    #             label_zeros = torch.zeros(batch_size, dtype=torch.long, device=device)
    #             label_ones = torch.ones(batch_size, dtype=torch.long, device=device)
    #             label_twos = 2 * torch.ones(batch_size, dtype=torch.long, device=device)
    #             labels = torch.cat([label_zeros, label_ones, label_twos], dim=0)
    #
    #             for k in real_inputs.keys():
    #                 discriminator = getattr(self.xz_discriminators, k)
    #
    #                 real_x = real_inputs[k][0]
    #                 enc_z = gen_inputs[k][1]
    #
    #                 dec_x = gen_inputs[k][0]
    #                 real_z = real_inputs[k][1]
    #
    #                 # shuffle x and z together
    #                 real_x_shuffled, enc_z_shuffled = utils.permute_dim([real_x, enc_z], dim=0)
    #                 enc_s_shuffled = enc_z_shuffled[:, :-self.content_dim]
    #
    #                 real_z_shuffled = utils.permute_dim(real_z, dim=0)
    #                 real_c_shuffled = real_z_shuffled[:, -self.content_dim:]
    #
    #                 cs_shuffled = torch.cat([enc_s_shuffled, real_c_shuffled], dim=1)
    #
    #                 # q(x, s, c)
    #                 dis_joint1 = discriminator(real_x, enc_z)
    #                 # p(x, s, c)
    #                 dis_joint2 = discriminator(dec_x, real_z)
    #                 # q(x, s) p(c)
    #                 dis_joint3 = discriminator(real_x_shuffled, cs_shuffled)
    #
    #                 losses['{}_dis_joint'.format(k)] = F.cross_entropy(torch.cat([dis_joint1,
    #                                                                               dis_joint2,
    #                                                                               dis_joint3], dim=0), labels)
    #                 # losses['{}_dis_x'.format(k)] = torch.mean(F.softplus(-dis_x1)) + torch.mean(F.softplus(dis_x2))
    #                 # losses['{}_dis_z'.format(k)] = F.cross_entropy(torch.cat([dix_z1,
    #                 #                                                           dix_z2,
    #                 #                                                           dix_z3], dim=0), labels)
    #     else:
    #         if joint:
    #             assert len(real_inputs.items()) == self.n_modalities
    #
    #             score_q = self.joint_discriminator(*[real_inputs[k][0] for k in sorted(real_inputs.keys())])
    #             score_p = self.joint_discriminator(*[gen_inputs[k][0] for k in sorted(gen_inputs.keys())])
    #
    #             def joint_score(inputs, use_q=True):
    #                 scores_xz = []
    #                 # make sure keys are sorted
    #                 for k in sorted(inputs.keys()):
    #                     discriminator = getattr(self.xz_discriminators, k)
    #                     x = inputs[k][0]
    #                     z = inputs[k][1]
    #
    #                     score = discriminator(x, z)
    #                     scores_xz.append(score)
    #
    #                 # avoid calculating joint score multiple times by binding to outer scope directly
    #                 score_joint = score_q if use_q else score_p
    #
    #                 scores = []
    #                 for s_xz in scores_xz:
    #                     # q(x, s, c) / (q(x, s) p(c))
    #                     s = s_xz[:, 0] - s_xz[:, 2]
    #                     scores.append(s.unsqueeze(-1))
    #
    #                 score = torch.sum(torch.stack([(s_xz[:, 1] - s_xz[:, 2]).unsqueeze(-1)  # p(x, s, c) / q(x, s) p(c)
    #                                                for s_xz in scores_xz], dim=0), dim=0) - score_joint
    #
    #                 scores.append(score)
    #
    #                 return torch.cat(scores, dim=1)
    #
    #             last_scores = []
    #             labels = []
    #             for k in sorted(real_inputs.keys()):
    #                 # real_inputs: key -> [x, z]
    #                 curr_inputs = real_inputs.copy()
    #                 content = gen_inputs[k][1][:, -self.content_dim:]
    #                 for k2, v2 in curr_inputs.items():
    #                     style = gen_inputs[k2][1][:, :-self.content_dim]
    #                     v2[1] = torch.cat([style, content], dim=1)
    #
    #                 last_scores.append(joint_score(curr_inputs, use_q=True))
    #                 labels.append(len(labels) * label_ones)
    #
    #             curr_inputs = gen_inputs.copy()
    #             for k, v in curr_inputs.items():
    #                 v[1] = real_inputs[k][1]
    #
    #             last_scores.append(joint_score(curr_inputs, use_q=False))
    #             labels.append(len(labels) * label_ones)
    #
    #             adv_losses = []
    #             for i in range(len(labels) - 1):
    #                 # shift labels
    #                 labels.insert(0, labels.pop())
    #                 adv_losses.append(F.cross_entropy(torch.cat(last_scores, dim=0),
    #                                                   torch.cat(labels, dim=0)))
    #             losses['joint'] = torch.mean(torch.stack(adv_losses, dim=0), dim=0)
    #
    #             if self.lambda_c_rec > 0.0:
    #                 z_rec_dist_params = []
    #
    #                 for k in real_inputs.keys():
    #                     encoder = getattr(self.encoders, k).module
    #                     z_rec_dist_param = encoder(gen_inputs[k][0])
    #                     z_rec_dist_params.append(z_rec_dist_param)
    #
    #                 z_joint = utils.reparameterize(utils.joint_posterior(*z_rec_dist_params, average=True))
    #
    #                 c_real = list(real_inputs.values())[0][1][:, -self.content_dim:]
    #                 c_rec = z_joint[:, -self.content_dim:]
    #                 c_rec_loss = (c_rec - c_real).square().mean()
    #                 losses['joint_c_rec'] = self.lambda_c_rec * c_rec_loss
    #
    #             if self.lambda_x_rec > 0.0:
    #                 z_joint = utils.reparameterize(utils.joint_posterior(*[v[2] for v in gen_inputs.values()],
    #                                                                      average=True))
    #                 c_joint = z_joint[:, -self.content_dim:]
    #
    #                 for k in real_inputs.keys():
    #                     decoder = getattr(self.decoders, k)
    #
    #                     x_real = real_inputs[k][0]
    #                     x_rec = decoder(torch.cat([gen_inputs[k][1][:, :-self.content_dim], c_joint], dim=1))
    #
    #                     x_rec_loss = (x_rec - x_real).square().mean()
    #                     losses['{}_x_rec'.format(k)] = self.lambda_x_rec * x_rec_loss
    #         else:
    #             for k in real_inputs.keys():
    #                 discriminator = getattr(self.xz_discriminators, k)
    #
    #                 real_x = real_inputs[k][0]
    #                 enc_z = gen_inputs[k][1]
    #
    #                 dec_x = gen_inputs[k][0]
    #                 real_z = real_inputs[k][1]
    #
    #                 # q(x, s, c) / p(x, s, c)
    #                 dis_real = discriminator(real_x, enc_z)
    #                 dis_real = dis_real[:, 0] - dis_real[:, 1]
    #                 dis_fake = discriminator(dec_x, real_z)
    #                 dis_fake = dis_fake[:, 0] - dis_fake[:, 1]
    #                 loss = torch.mean(F.softplus(dis_real)) + torch.mean(F.softplus(-dis_fake))
    #                 losses[k] = self.lambda_unimodal * loss
    #                 # _, _, dis_z = discriminator(real_x, enc_z, return_xz=True)
    #                 # dis_z = dis_z[:, 1] - dis_z[:, 0]
    #                 # _, dis_x, _ = discriminator(dec_x, real_z, return_xz=True)
    #                 #
    #                 # loss_x = torch.mean(F.softplus(-dis_x))
    #                 # loss_z = torch.mean(F.softplus(-dis_z))
    #                 # losses[k] = self.lambda_unimodal * loss_x + self.lambda_unimodal * loss_z
    #
    #             if self.lambda_s_rec > 0.0:
    #                 for k in real_inputs.keys():
    #                     encoder = getattr(self.encoders, k)
    #                     z_rec = encoder(gen_inputs[k][0])
    #
    #                     s_real = real_inputs[k][1][:, :-self.content_dim]
    #                     s_rec = z_rec[:, :-self.content_dim]
    #
    #                     s_rec_loss = (s_rec - s_real).square().mean()
    #
    #                     losses['{}_s_rec'.format(k)] = self.lambda_s_rec * s_rec_loss
    #
    #     return losses
