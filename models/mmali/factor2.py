import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class FactorModelDoubleSemi(nn.Module):
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

        for modality_key in self.sorted_keys:
            # q(x, s, c) : p(x, s, c)
            score = scores[modality_key][0]
            joint_score.append(score - score_sum)

        return torch.cat(joint_score, dim=1)

    def forward(self, real_inputs, train_d=True, joint=False, progress=None):
        return self.forward_jsd(real_inputs, train_d=train_d, joint=joint, progress=progress)

    def forward_jsd(self, real_inputs, train_d=True, joint=False, progress=None):
        self.generators.requires_grad_(not train_d)
        self.discriminators.requires_grad_(train_d)

        batch_size = list(real_inputs.values())[0]['x'].size(0)
        device = list(real_inputs.values())[0]['x'].device
        # label_zeros = torch.zeros(batch_size, dtype=torch.long, device=device)
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

                    # q(x, s, c) : q(x, s) p(c)
                    dis_real = discriminator[1](real_x, enc_z)
                    dis_fake = discriminator[1](real_x_shuffled, cs_shuffled)
                    losses['{}_1'.format(modality_key)] = torch.mean(F.softplus(-dis_real)) + \
                                                          torch.mean(F.softplus(dis_fake))
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
                score_q = self.joint_discriminator(*[real_inputs[k]['x'] for k in self.sorted_keys])
                for modality_key in self.sorted_keys:
                    curr_inputs = {k: {} for k in self.sorted_keys}
                    content = gen_inputs[modality_key]['z'][:, -self.content_dim:]

                    for modality_key2 in self.sorted_keys:
                        style = gen_inputs[modality_key2]['z'][:, :-self.content_dim]

                        curr_inputs[modality_key2]['x'] = real_inputs[modality_key2]['x']
                        curr_inputs[modality_key2]['z'] = torch.cat([style, content], dim=1)

                    dis_score = self.calc_joint_score(curr_inputs, score_q)
                    adv_losses = [F.cross_entropy(dis_score, i * label_ones)
                                  for i in range(dis_score.size(1)) if i != label_value]
                    # TODO: torch.sum or torch.mean?
                    losses['joint_q{}'.format(label_value)] = torch.mean(torch.stack(adv_losses, dim=0), dim=0)

                    label_value += 1

                score_p = self.joint_discriminator(*[gen_inputs[k]['x'] for k in self.sorted_keys])
                curr_inputs = {k: {} for k in self.sorted_keys}
                for modality_key in self.sorted_keys:
                    curr_inputs[modality_key]['x'] = gen_inputs[modality_key]['x']
                    curr_inputs[modality_key]['z'] = real_inputs[modality_key]['z']

                dis_score = self.calc_joint_score(curr_inputs, score_p)
                adv_losses = [F.cross_entropy(dis_score, i * label_ones)
                              for i in range(dis_score.size(1)) if i != label_value]
                losses['joint_p{}'.format(label_value)] = torch.mean(torch.stack(adv_losses, dim=0), dim=0)

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
                label_value = self.n_modalities + 1
                for modality_key in self.sorted_keys:
                    curr_inputs = {k: {} for k in self.sorted_keys}
                    curr_inputs[modality_key]['x'] = real_inputs[modality_key]['x']
                    curr_inputs[modality_key]['z'] = gen_inputs[modality_key]['z']

                    enc_c = gen_inputs[modality_key]['z'][:, -self.content_dim:]
                    for other_key in self.sorted_keys:
                        if other_key == modality_key:
                            continue

                        decoder = getattr(self.decoders, other_key)
                        real_s = utils.permute_dim(real_inputs[other_key]['z'][:, :-self.content_dim], dim=0)
                        combined_z = torch.cat([real_s, enc_c], dim=1)

                        rec_x = decoder(combined_z)
                        curr_inputs[other_key]['x'] = rec_x
                        curr_inputs[other_key]['z'] = combined_z

                    score_joint = self.joint_discriminator(*[curr_inputs[k]['x'] for k in self.sorted_keys])
                    dis_score = self.calc_joint_score(curr_inputs, score_joint)

                    adv_losses = [F.cross_entropy(dis_score, i * label_ones)
                                  for i in range(dis_score.size(1)) if i != label_value]

                    losses['joint_r{}'.format(label_value)] = torch.mean(torch.stack(adv_losses, dim=0), dim=0)

                    label_value += 1

                for modality_key in self.sorted_keys:
                    discriminator = getattr(self.xz_discriminators, modality_key)[0]

                    real_x = real_inputs[modality_key]['x']
                    enc_z = gen_inputs[modality_key]['z']

                    dec_x = gen_inputs[modality_key]['x']
                    real_z = real_inputs[modality_key]['z']

                    dis_real = discriminator(real_x, enc_z)
                    dis_fake = discriminator(dec_x, real_z)
                    losses['{}_real_fake'.format(modality_key)] = self.lambda_unimodal * (
                            torch.mean(F.softplus(dis_real)) + torch.mean(F.softplus(-dis_fake)))

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

    #
    # def forward_mi_nondet(self, real_inputs, train_d=True, joint=False, progress=None):
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
    #             self.encoders.requires_grad_(True)
    #
    #             # make sure the order of sorted keys matches the input order of joint discriminator
    #             joint_inputs = [getattr(self.encoders, k)(real_inputs[k][0])
    #                             for k in sorted(real_inputs.keys())]
    #             shuffled_inputs = [utils.permute_dim(x) for x in joint_inputs]
    #
    #             dis_real = self.joint_discriminator(*joint_inputs)
    #             dis_fake = self.joint_discriminator(*shuffled_inputs)
    #
    #             losses['joint'] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
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
    #                 # shuffle x and z together
    #                 real_x_shuffled, enc_z_shuffled = utils.permute_dim([real_x, enc_z], dim=0)
    #                 enc_s_shuffled = enc_z_shuffled[:, :-self.content_dim]
    #
    #                 real_z_shuffled = utils.permute_dim(real_z, dim=0)
    #                 real_c_shuffled = real_z_shuffled[:, -self.content_dim:]
    #
    #                 cs_shuffled = torch.cat([enc_s_shuffled, real_c_shuffled], dim=1)
    #
    #                 # q(x, s, c) / p(x, s, c)
    #                 dis_real = discriminator[0](real_x, enc_z)
    #                 dis_fake = discriminator[0](dec_x, real_z)
    #                 losses['{}_0'.format(k)] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #
    #                 # p(x, s, c) / q(x, s) p(c)
    #                 dis_real = discriminator[1](dec_x, real_z)
    #                 dis_fake = discriminator[1](real_x_shuffled, cs_shuffled)
    #                 losses['{}_1'.format(k)] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #     else:
    #         if joint:
    #             assert len(real_inputs.items()) == self.n_modalities
    #
    #             self.encoders.requires_grad_(False)
    #             score_q = self.joint_discriminator(*[getattr(self.encoders, k)(real_inputs[k][0])
    #                                                  for k in sorted(real_inputs.keys())])
    #             score_p = self.joint_discriminator(*[getattr(self.encoders, k)(gen_inputs[k][0])
    #                                                  for k in sorted(gen_inputs.keys())])
    #             self.encoders.requires_grad_(True)
    #
    #             def joint_score(inputs, use_q=True):
    #                 scores_xz = []
    #                 # make sure keys are sorted
    #                 for k in sorted(inputs.keys()):
    #                     discriminator = getattr(self.xz_discriminators, k)
    #                     x = inputs[k][0]
    #                     z = inputs[k][1]
    #
    #                     score1 = discriminator[0](x, z)  # q(x, s, c) / p(x, s, c)
    #                     score2 = discriminator[1](x, z)  # p(x, s, c) / q(x, s) p(c)
    #                     scores_xz.append([score1, score2])
    #
    #                 # avoid calculating joint score multiple times by binding to outer scope directly
    #                 score_joint = score_q if use_q else score_p
    #
    #                 scores = []
    #                 for s_xz in scores_xz:
    #                     # q(x, s, c) / (q(x, s) p(c))
    #                     scores.append(s_xz[0] + s_xz[1])
    #
    #                 score = torch.sum(torch.stack([s_xz[1]  # p(x, s, c) / q(x, s) p(c)
    #                                                for s_xz in scores_xz], dim=0), dim=0) - score_joint
    #
    #                 scores.append(score)
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
    #                 dis_real = discriminator[0](real_x, enc_z)
    #                 dis_fake = discriminator[0](dec_x, real_z)
    #
    #                 loss = torch.mean(F.softplus(dis_real)) + torch.mean(F.softplus(-dis_fake))
    #                 losses[k] = self.lambda_unimodal * loss
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
    # def forward1(self, real_inputs, train_d=True, joint=False, progress=None):
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
    #                 encoder = getattr(self.encoders, k)
    #                 decoder = getattr(self.decoders, k)
    #                 enc_z = encoder(v[0])
    #                 dec_x = decoder(v[1])
    #                 gen_inputs[k] = [dec_x, enc_z]
    #
    #     losses = {}
    #     if train_d:
    #         if joint:
    #             # all modalities are available
    #             assert len(real_inputs.items()) == self.n_modalities
    #
    #             # make sure the order of sorted keys matches the input order of joint discriminator
    #             joint_inputs = [real_inputs[k][0] for k in sorted(real_inputs.keys())]
    #             shuffled_inputs = [utils.permute_dim(real_inputs[k][2], dim=0) for k in sorted(real_inputs.keys())]
    #
    #             dis_real = self.joint_discriminator(*joint_inputs)
    #             dis_fake = self.joint_discriminator(*shuffled_inputs)
    #
    #             losses['joint'] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
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
    #                 # shuffle x and z together
    #                 real_x_shuffled, enc_z_shuffled = utils.permute_dim([real_x, enc_z], dim=0)
    #                 enc_s_shuffled = enc_z_shuffled[:, :-self.content_dim]
    #
    #                 real_z_shuffled = utils.permute_dim(real_z, dim=0)
    #                 real_c_shuffled = real_z_shuffled[:, -self.content_dim:]
    #
    #                 cs_shuffled = torch.cat([enc_s_shuffled, real_c_shuffled], dim=1)
    #
    #                 # if self.lambda_gp[0] > 0.0:
    #                 #     real_x.requires_grad_(True)
    #                 #     enc_z.requires_grad_(True)
    #                 #     dec_x.requires_grad_(True)
    #                 #     real_z.requires_grad_(True)
    #
    #                 # q(x, s, c) / p(x, s, c)
    #                 dis_real = discriminator[0](real_x, enc_z)
    #                 dis_fake = discriminator[0](dec_x, real_z)
    #                 losses['{}_0'.format(k)] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #                 # if self.lambda_gp[0] > 0.0:
    #                 #     gamma = self.lambda_gp[0] * self.lambda_gp[1] ** progress
    #                 #     losses['{}_0_gp'.format(k)] = utils.calc_gradient_penalty_jsd(
    #                 #         dis_real, [real_x, enc_z], dis_fake, [dec_x, real_z], gamma=gamma
    #                 #     )
    #
    #                 # dec_x.requires_grad_(True)
    #                 # real_z.requires_grad_(True)
    #                 # real_x_shuffled.requires_grad_(True)
    #                 # cs_shuffled.requires_grad_(True)
    #
    #                 # p(x, s, c) / q(x, s) p(c)
    #                 dis_real = discriminator[1](dec_x, real_z)
    #                 dis_fake = discriminator[1](real_x_shuffled, cs_shuffled)
    #                 losses['{}_1'.format(k)] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #                 # losses['{}_1_gp'.format(k)] = utils.calc_gradient_penalty_jsd(
    #                 #     dis_real, [dec_x, real_z],
    #                 #     dis_fake, [real_x_shuffled, cs_shuffled], gamma=0.1
    #                 # )
    #
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
    #                     score1 = discriminator[0](x, z)  # q(x, s, c) / p(x, s, c)
    #                     score2 = discriminator[1](x, z)  # p(x, s, c) / q(x, s) p(c)
    #                     scores_xz.append([score1, score2])
    #
    #                 # avoid calculating joint score multiple times by binding to outer scope directly
    #                 score_joint = score_q if use_q else score_p
    #
    #                 scores = []
    #                 for s_xz in scores_xz:
    #                     # q(x, s, c) / (q(x, s) p(c))
    #                     scores.append(s_xz[0] + s_xz[1])
    #
    #                 score = torch.sum(torch.stack([s_xz[1]  # p(x, s, c) / q(x, s) p(c)
    #                                                for s_xz in scores_xz], dim=0), dim=0) - score_joint
    #
    #                 scores.append(score)
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
    #                 #  shift labels
    #                 labels.insert(0, labels.pop())
    #                 adv_losses.append(F.cross_entropy(torch.cat(last_scores, dim=0),
    #                                                   torch.cat(labels, dim=0)))
    #             losses['joint'] = torch.mean(torch.stack(adv_losses, dim=0), dim=0)
    #
    #             if self.lambda_z_rec > 0.0:
    #                 z_rec_dist_params = []
    #                 # every modality should be available
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
    #                 losses['joint_c_rec'] = self.lambda_z_rec * c_rec_loss
    #
    #             # if self.lambda_x_rec > 0.0:
    #             #     for k in real_inputs.keys():
    #             #         decoder = getattr(self.decoders, k)
    #             #         loss = (real_inputs[k][0] - decoder(gen_inputs[k][1])).square().mean()
    #             #         losses['{}_x_rec'.format(k)] = self.lambda_x_rec * loss
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
    #                 dis_real = discriminator[0](real_x, enc_z)
    #                 dis_fake = discriminator[0](dec_x, real_z)
    #
    #                 loss = torch.mean(F.softplus(dis_real)) + torch.mean(F.softplus(-dis_fake))
    #                 losses[k] = self.lambda_unimodal * loss
    #                 # if self.lambda_unimodal[0] > 0.0:
    #                 #     lambda_unimodal = self.lambda_unimodal[0] * self.lambda_unimodal[1] ** progress
    #                 #     losses[k] = lambda_unimodal * loss
    #                 # else:
    #                 #     losses[k] = loss
    #
    #             if self.lambda_x_rec > 0.0:
    #                 for k in real_inputs.keys():
    #                     decoder = getattr(self.decoders, k)
    #                     loss = (real_inputs[k][0] - decoder(gen_inputs[k][1])).square().mean()
    #                     losses['{}_x_rec'.format(k)] = self.lambda_x_rec * loss
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
    #                 if self.lambda_gp[0] > 0.0:
    #                     real_x.requires_grad_(True)
    #                     enc_z.requires_grad_(True)
    #                     dec_x.requires_grad_(True)
    #                     real_z.requires_grad_(True)
    #
    #                 # q(x, s, c) / p(x, s, c)
    #                 dis_real = discriminator[0](real_x, enc_z)
    #                 dis_fake = discriminator[0](dec_x, real_z)
    #                 losses['{}_0'.format(k)] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #                 if self.lambda_gp[0] > 0.0:
    #                     gamma = self.lambda_gp[0] * self.lambda_gp[1] ** progress
    #                     losses['{}_0_gp'.format(k)] = utils.calc_gradient_penalty_jsd(
    #                         dis_real, [real_x, enc_z], dis_fake, [dec_x, real_z], gamma=gamma
    #                     )
    #
    #                 # dec_x.requires_grad_(True)
    #                 # real_z.requires_grad_(True)
    #                 # real_x_shuffled.requires_grad_(True)
    #                 # cs_shuffled.requires_grad_(True)
    #
    #                 # p(x, s, c) / q(x, s) p(c)
    #                 dis_real = discriminator[1](dec_x, real_z)
    #                 dis_fake = discriminator[1](real_x_shuffled, cs_shuffled)
    #                 losses['{}_1'.format(k)] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #                 # losses['{}_1_gp'.format(k)] = utils.calc_gradient_penalty_jsd(
    #                 #     dis_real, [dec_x, real_z],
    #                 #     dis_fake, [real_x_shuffled, cs_shuffled], gamma=0.1
    #                 # )
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
    #                     score1 = discriminator[0](x, z)  # q(x, s, c) / p(x, s, c)
    #                     score2 = discriminator[1](x, z)  # p(x, s, c) / q(x, s) p(c)
    #                     scores_xz.append([score1, score2])
    #
    #                 # avoid calculating joint score multiple times by binding to outer scope directly
    #                 score_joint = score_q if use_q else score_p
    #
    #                 scores = []
    #                 for s_xz in scores_xz:
    #                     # q(x, s, c) / (q(x, s) p(c))
    #                     scores.append(s_xz[0] + s_xz[1])
    #
    #                 score = torch.sum(torch.stack([s_xz[1]  # p(x, s, c) / q(x, s) p(c)
    #                                                for s_xz in scores_xz], dim=0), dim=0) - score_joint
    #
    #                 scores.append(score)
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
    #                 dis_real = discriminator[0](real_x, enc_z)
    #                 dis_fake = discriminator[0](dec_x, real_z)
    #
    #                 loss = torch.mean(F.softplus(dis_real)) + torch.mean(F.softplus(-dis_fake))
    #                 losses[k] = self.lambda_unimodal * loss
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
    #
    # def forward_new(self, real_inputs, train_d=True, joint=False, progress=None):
    #     self.generators.requires_grad_(not train_d)
    #     self.discriminators.requires_grad_(train_d)
    #
    #     # real_inputs: key -> [real_x, real_z] or key -> [paired_x, unpaired_x]
    #     batch_size = list(real_inputs.values())[0][0].size(0)
    #     device = list(real_inputs.values())[0][0].device
    #     label_ones = torch.ones(batch_size, dtype=torch.long, device=device)
    #
    #     with torch.set_grad_enabled(not train_d):
    #         # inputs: key -> [x, z] where z = style + content
    #         gen_inputs = {}
    #         for k, v in real_inputs.items():
    #             encoder = getattr(self.encoders, k).module
    #             decoder = getattr(self.decoders, k)
    #             enc_z_dist_param = encoder(v[0])
    #             enc_z = utils.reparameterize(enc_z_dist_param)
    #             dec_x = decoder(v[1])
    #             gen_inputs[k] = [dec_x, enc_z, enc_z_dist_param]
    #
    #     losses = {}
    #     if train_d:
    #         if joint:
    #             # all modalities should be available
    #             assert len(real_inputs.items()) == self.n_modalities
    #
    #             # make sure the order of sorted keys matches the input order of joint discriminator
    #             joint_inputs = [real_inputs[k][0] for k in sorted(real_inputs.keys())]
    #             shuffled_inputs = [utils.permute_dim(real_inputs[k][2], dim=0) for k in sorted(real_inputs.keys())]
    #
    #             dis_real = self.joint_discriminator(*joint_inputs)
    #             dis_fake = self.joint_discriminator(*shuffled_inputs)
    #
    #             losses['joint'] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
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
    #                 # p(x, s, c) / q(x, s) p(c)
    #                 dis_real = discriminator[1](dec_x, real_z)
    #                 dis_fake = discriminator[1](real_x_shuffled, cs_shuffled)
    #                 losses['{}_1'.format(k)] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #
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
    #                 dis_real, dis_x_real, dis_z_fake = discriminator[0](real_x, enc_z, return_xz=True)
    #                 dis_fake, dis_x_fake, dis_z_real = discriminator[0](dec_x, real_z, return_xz=True)
    #                 losses['{}_0_joint'.format(k)] = torch.mean(F.softplus(-dis_real)) \
    #                                                  + torch.mean(F.softplus(dis_fake))
    #                 losses['{}_0_x'.format(k)] = torch.mean(F.softplus(-dis_x_real)) \
    #                                              + torch.mean(F.softplus(dis_x_fake))
    #                 losses['{}_0_z'.format(k)] = torch.mean(F.softplus(-dis_z_real)) \
    #                                              + torch.mean(F.softplus(dis_z_fake))
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
    #                     score1 = discriminator[0](x, z)  # q(x, s, c) / p(x, s, c)
    #                     score2 = discriminator[1](x, z)  # p(x, s, c) / q(x, s) p(c)
    #                     scores_xz.append([score1, score2])
    #
    #                 # avoid calculating joint score multiple times by binding to outer scope directly
    #                 score_joint = score_q if use_q else score_p
    #
    #                 scores = []
    #                 for s_xz in scores_xz:
    #                     # q(x, s, c) / (q(x, s) p(c))
    #                     scores.append(s_xz[0] + s_xz[1])
    #
    #                 score = torch.sum(torch.stack([s_xz[1]  # p(x, s, c) / q(x, s) p(c)
    #                                                for s_xz in scores_xz], dim=0), dim=0) - score_joint
    #
    #                 scores.append(score)
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
    #                 enc_z = gen_inputs[k][1]
    #
    #                 dec_x = gen_inputs[k][0]
    #
    #                 # q(x, s, c) / p(x, s, c)
    #                 dis_z_fake = discriminator[0](x=None, z=enc_z)
    #                 dis_x_fake = discriminator[0](x=dec_x, z=None)
    #
    #                 loss_z = torch.mean(F.softplus(-dis_z_fake))
    #                 loss_x = torch.mean(F.softplus(-dis_x_fake))
    #                 losses['{}_z'.format(k)] = self.lambda_unimodal * loss_z
    #                 losses['{}_x'.format(k)] = self.lambda_unimodal * loss_x
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
    #
    # def forward_mi_det(self, real_inputs, train_d=True, joint=False, progress=None):
    #     # self.encoders.requires_grad_(train_d)
    #     # self.decoders.requires_grad_(not train_d)
    #     # self.discriminators.requires_grad_(train_d)
    #
    #     # real_inputs: key -> [real_x, real_z] or key -> [paired_x, unpaired_x]
    #     batch_size = list(real_inputs.values())[0][0].size(0)
    #     device = list(real_inputs.values())[0][0].device
    #     label_ones = torch.ones(batch_size, dtype=torch.long, device=device)
    #
    #     # train encoders as part of discriminators
    #     # how about unimodal training here? : currently train encoder via discriminator and reconstruction
    #     # if not train_d and not joint:
    #     #     self.encoders.requires_grad_(True)
    #     # else:
    #     #     self.encoders.requires_grad_(train_d)
    #
    #     if train_d:
    #         self.decoders.requires_grad_(False)
    #         self.discriminators.requires_grad_(True)
    #         if joint:
    #             self.encoders.requires_grad_(True)
    #         else:
    #             self.encoders.requires_grad_(False)
    #     else:
    #         self.decoders.requires_grad_(True)
    #         self.discriminators.requires_grad_(False)
    #         if joint:
    #             self.encoders.requires_grad_(False)
    #         else:
    #             # train unimodal encoders adversarially?
    #             self.encoders.requires_grad_(True)
    #
    #     if not (train_d and joint):
    #         gen_inputs = {}
    #         for k, v in real_inputs.items():
    #             encoder = getattr(self.encoders, k)
    #             decoder = getattr(self.decoders, k)
    #             enc_z = encoder(v[0])
    #             dec_x = decoder(v[1])
    #             gen_inputs[k] = [dec_x, enc_z]
    #
    #     losses = {}
    #     if train_d:
    #         if joint:
    #             # all modalities should be available
    #             assert len(real_inputs.items()) == self.n_modalities
    #
    #             # discriminator as encoder
    #             self.encoders.requires_grad_(True)
    #
    #             # make sure the order of sorted keys matches the input order of joint discriminator
    #             joint_inputs = [getattr(self.encoders, k)(real_inputs[k][0])
    #                             for k in sorted(real_inputs.keys())]
    #             shuffled_inputs = [utils.permute_dim(x) for x in joint_inputs]
    #
    #             dis_real = self.joint_discriminator(*joint_inputs)
    #             dis_fake = self.joint_discriminator(*shuffled_inputs)
    #
    #             losses['joint'] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
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
    #                 # shuffle x and z together
    #                 real_x_shuffled, enc_z_shuffled = utils.permute_dim([real_x, enc_z], dim=0)
    #                 enc_s_shuffled = enc_z_shuffled[:, :-self.content_dim]
    #
    #                 real_z_shuffled = utils.permute_dim(real_z, dim=0)
    #                 real_c_shuffled = real_z_shuffled[:, -self.content_dim:]
    #
    #                 cs_shuffled = torch.cat([enc_s_shuffled, real_c_shuffled], dim=1)
    #
    #                 # q(x, s, c) / p(x, s, c)
    #                 dis_real = discriminator[0](real_x, enc_z)
    #                 dis_fake = discriminator[0](dec_x, real_z)
    #                 losses['{}_0'.format(k)] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #
    #                 # p(x, s, c) / q(x, s) p(c)
    #                 dis_real = discriminator[1](dec_x, real_z)
    #                 dis_fake = discriminator[1](real_x_shuffled, cs_shuffled)
    #                 losses['{}_1'.format(k)] = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    #     else:
    #         if joint:
    #             assert len(real_inputs.items()) == self.n_modalities
    #
    #             self.encoders.requires_grad_(False)
    #             score_q = self.joint_discriminator(*[getattr(self.encoders, k)(real_inputs[k][0])
    #                                                  for k in sorted(real_inputs.keys())])
    #             score_p = self.joint_discriminator(*[getattr(self.encoders, k)(gen_inputs[k][0])
    #                                                  for k in sorted(gen_inputs.keys())])
    #             self.encoders.requires_grad_(True)
    #
    #             def joint_score(inputs, use_q=True):
    #                 scores_xz = []
    #                 # make sure keys are sorted
    #                 for k in sorted(inputs.keys()):
    #                     discriminator = getattr(self.xz_discriminators, k)
    #                     x = inputs[k][0]
    #                     z = inputs[k][1]
    #
    #                     score1 = discriminator[0](x, z)  # q(x, s, c) / p(x, s, c)
    #                     score2 = discriminator[1](x, z)  # p(x, s, c) / q(x, s) p(c)
    #                     scores_xz.append([score1, score2])
    #
    #                 # avoid calculating joint score multiple times by binding to outer scope directly
    #                 score_joint = score_q if use_q else score_p
    #                 # score_joint = score_p
    #
    #                 scores = []
    #                 for s_xz in scores_xz:
    #                     # q(x, s, c) / (q(x, s) p(c))
    #                     scores.append(s_xz[0] + s_xz[1])
    #
    #                 score = torch.sum(torch.stack([s_xz[1]  # p(x, s, c) / q(x, s) p(c)
    #                                                for s_xz in scores_xz], dim=0), dim=0) - score_joint
    #
    #                 scores.append(score)
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
    #             # if self.lambda_c_rec > 0.0:
    #             #     z_rec_dist_params = []
    #             #
    #             #     for k in real_inputs.keys():
    #             #         encoder = getattr(self.encoders, k).module
    #             #         z_rec_dist_param = encoder(gen_inputs[k][0])
    #             #         z_rec_dist_params.append(z_rec_dist_param)
    #             #
    #             #     z_joint = utils.reparameterize(utils.joint_posterior(*z_rec_dist_params, average=True))
    #             #
    #             #     c_real = list(real_inputs.values())[0][1][:, -self.content_dim:]
    #             #     c_rec = z_joint[:, -self.content_dim:]
    #             #     c_rec_loss = (c_rec - c_real).square().mean()
    #             #     losses['joint_c_rec'] = self.lambda_c_rec * c_rec_loss
    #
    #             # if self.lambda_x_rec > 0.0:
    #             #     z_joint = utils.reparameterize(utils.joint_posterior(*[v[2] for v in gen_inputs.values()],
    #             #                                                          average=True))
    #             #     c_joint = z_joint[:, -self.content_dim:]
    #             #
    #             #     for k in real_inputs.keys():
    #             #         decoder = getattr(self.decoders, k)
    #             #
    #             #         x_real = real_inputs[k][0]
    #             #         x_rec = decoder(torch.cat([gen_inputs[k][1][:, :-self.content_dim], c_joint], dim=1))
    #             #
    #             #         x_rec_loss = (x_rec - x_real).square().mean()
    #             #         losses['{}_x_rec'.format(k)] = self.lambda_x_rec * x_rec_loss
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
    #                 dis_real = discriminator[0](real_x, enc_z)
    #                 dis_fake = discriminator[0](dec_x, real_z)
    #
    #                 loss = torch.mean(F.softplus(dis_real)) + torch.mean(F.softplus(-dis_fake))
    #                 # loss = torch.mean(F.softplus(-dis_fake))
    #                 losses[k] = self.lambda_unimodal * loss
    #
    #             if self.lambda_s_rec > 0.0:
    #                 self.encoders.requires_grad_(True)
    #                 for k in real_inputs.keys():
    #                     encoder = getattr(self.encoders, k)
    #                     z_rec = encoder(gen_inputs[k][0])
    #
    #                     s_real = real_inputs[k][1]
    #                     s_rec = z_rec
    #
    #                     s_rec_loss = (s_rec - s_real).square().mean()
    #
    #                     losses['{}_s_rec'.format(k)] = self.lambda_s_rec * s_rec_loss
    #
    #             if self.lambda_x_rec > 0.0:
    #                 self.encoders.requires_grad_(True)
    #                 for k in real_inputs.keys():
    #                     decoder = getattr(self.decoders, k)
    #
    #                     x_real = real_inputs[k][0]
    #                     x_rec = decoder(gen_inputs[k][1])
    #
    #                     x_rec_loss = (x_rec - x_real).square().mean()
    #                     losses['{}_x_rec'.format(k)] = self.lambda_x_rec * x_rec_loss
    #
    #     return losses
