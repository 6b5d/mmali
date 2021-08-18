import math
from collections import Counter, OrderedDict
from itertools import chain, combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from scipy.linalg import eig
from skimage.filters import threshold_yen as threshold


def permute_dim(input, dim=0):
    if torch.is_tensor(input):
        index = torch.randperm(input.size(dim)).to(input.device)
        return torch.index_select(input, dim=dim, index=index)
    else:
        index = torch.randperm(input[0].size(dim)).to(input[0].device)
        return [torch.index_select(x, dim=dim, index=index) for x in input]


def init_param_normal(model, mean=0.0, std=0.02):
    param_count = 0
    for m in model.modules():
        # Linear
        # Conv1d/Conv2d/ConvTranspose1d/ConvTranspose2d
        if isinstance(m, nn.Linear) or m.__class__.__name__.startswith('Conv'):
            init.normal_(m.weight.data, mean, std)
            if m.bias is not None:
                init.zeros_(m.bias.data)
            param_count += sum([p.data.nelement() for p in m.parameters()])

        # BatchNorm1d/BatchNorm2d/BatchNorm3d
        if m.__class__.__name__.startswith('BatchNorm'):
            init.ones_(m.weight.data)
            init.zeros_(m.bias.data)
            param_count += sum([p.data.nelement() for p in m.parameters()])

    print('{}\'s parameters initialized: {}'.format(model.__class__.__name__, param_count))


def product_of_experts(mu, logvar, average=True):
    # calculate PoE with LogSumExp trick
    N = mu.size(0)
    poe_logvar = -torch.logsumexp(-logvar, dim=0)
    w = torch.exp(-logvar + poe_logvar)
    poe_mu = torch.sum(w * mu, dim=0)
    if average:
        poe_logvar = poe_logvar + math.log(N)

    return torch.cat([poe_mu, poe_logvar], dim=1)


def joint_posterior(*Z, average=True):
    mu = []
    logvar = []
    for z in Z:
        ch = z.size(1) // 2
        mu.append(z[:, :ch])
        logvar.append(z[:, ch:])
    return product_of_experts(torch.stack(mu, dim=0), torch.stack(logvar, dim=0), average=average)


def reparameterize(z):
    c = z.size(1) // 2
    mu, logvar = z[:, :c], z[:, c:]

    std = torch.exp(0.5 * logvar)
    # std = torch.sqrt(F.softmax(logvar, dim=-1) * logvar.size(-1))
    sampled_z = torch.randn_like(mu)

    return sampled_z * std + mu


# make sure it is sorted as expected
def powerset(iterable, remove_null=True):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    start_idx = 1 if remove_null else 0
    return chain.from_iterable(combinations(s, r) for r in range(start_idx, len(s) + 1))


def decay_lr(opt, max_iter, start_iter, initial_lr):
    """Decay learning rate linearly till 0."""
    coeff = -initial_lr / (max_iter - start_iter)
    for pg in opt.param_groups:
        pg['lr'] += coeff


def calc_gradient_penalty(discriminator, decoder, z, enc_z, lambda_gp=5.):
    with torch.no_grad():
        alpha = torch.rand(enc_z.size(0), 1, device=enc_z.device)
        alpha = alpha.expand(enc_z.size())
        z_interpolates = alpha * enc_z + (1. - alpha) * z
        x_interpolates = decoder(z_interpolates)

    x_interpolates.requires_grad_(True)
    # z_interpolates.requires_grad_(True)

    disc_interpolates = discriminator(x_interpolates, z_interpolates)
    gradients_x, = torch.autograd.grad(outputs=disc_interpolates, inputs=[x_interpolates],
                                       grad_outputs=torch.ones_like(disc_interpolates),
                                       create_graph=True, retain_graph=True, only_inputs=True)
    gradients_x = torch.flatten(gradients_x, start_dim=1)
    # gradients_z = torch.flatten(gradients_z, start_dim=1)

    gradient_penalty = lambda_gp * (gradients_x.norm(2, dim=1).square()).mean()

    return gradient_penalty


def calc_gradient_penalty_jsd(logits_real, reals, logits_fake, fakes, gamma):
    gradients_real = torch.autograd.grad(logits_real, reals,
                                         grad_outputs=torch.ones_like(logits_real),
                                         create_graph=True,
                                         retain_graph=True,
                                         only_inputs=True)
    gradients_real = [torch.flatten(g, start_dim=1) for g in gradients_real]

    gradients_fake = torch.autograd.grad(logits_fake, fakes,
                                         grad_outputs=torch.ones_like(logits_fake),
                                         create_graph=True,
                                         retain_graph=True,
                                         only_inputs=True)
    gradients_fake = [torch.flatten(g, start_dim=1) for g in gradients_fake]

    prob_real = torch.flatten(torch.sigmoid(logits_real), start_dim=1)
    prob_fake = torch.flatten(torch.sigmoid(logits_fake), start_dim=1)

    gp = ((1. - prob_real).square() * sum([g.square().sum(dim=1) for g in gradients_real])).mean() \
         + (prob_fake.square() * sum([g.square().sum(dim=1) for g in gradients_fake])).mean()

    return 0.5 * gamma * gp


def calc_gaussian_entropy(logvar, dim=-1):
    return 0.5 * math.log(2 * math.pi * math.e) + 0.5 * torch.sum(logvar, dim=dim)


def calc_kl_divergence(mu0, logvar0, mu1=None, logvar1=None):
    if mu1 is None or logvar1 is None:
        KLD = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0)
    else:
        KLD = -0.5 * (
            torch.sum(1 - logvar0.exp() / logvar1.exp() - (mu0 - mu1).pow(2) / logvar1.exp() + logvar0 - logvar1))

    return KLD


def cca(views, k=None, eps=1e-12):
    """Compute (multi-view) CCA

    Args:
        views (list): list of views where each view `v_i` is of size `N x o_i`
        k (int): joint projection dimension | if None, find using Otsu
        eps (float): regulariser [default: 1e-12]

    Returns:
        correlations: correlations along each of the k dimensions
        projections: projection matrices for each view
    """
    V = len(views)  # number of views
    N = views[0].size(0)  # number of observations (same across views)
    os = [v.size(1) for v in views]
    kmax = np.min(os)
    ocum = np.cumsum([0] + os)
    os_sum = sum(os)
    A, B = np.zeros([os_sum, os_sum]), np.zeros([os_sum, os_sum])

    for i in range(V):
        v_i = views[i]
        v_i_bar = v_i - v_i.mean(0).expand_as(v_i)  # centered, N x o_i
        C_ij = (1.0 / (N - 1)) * torch.mm(v_i_bar.t(), v_i_bar)
        # A[ocum[i]:ocum[i + 1], ocum[i]:ocum[i + 1]] = C_ij
        B[ocum[i]:ocum[i + 1], ocum[i]:ocum[i + 1]] = C_ij
        for j in range(i + 1, V):
            v_j = views[j]  # N x o_j
            v_j_bar = v_j - v_j.mean(0).expand_as(v_j)  # centered
            C_ij = (1.0 / (N - 1)) * torch.mm(v_i_bar.t(), v_j_bar)
            A[ocum[i]:ocum[i + 1], ocum[j]:ocum[j + 1]] = C_ij
            A[ocum[j]:ocum[j + 1], ocum[i]:ocum[i + 1]] = C_ij.t()

    A[np.diag_indices_from(A)] += eps
    B[np.diag_indices_from(B)] += eps

    eigenvalues, eigenvectors = eig(A, B)
    # TODO: sanity check to see that all eigenvalues are e+0i
    idx = eigenvalues.argsort()[::-1]  # sort descending
    eigenvalues = eigenvalues[idx]  # arrange in descending order

    if k is None:
        t = threshold(eigenvalues.real[:kmax])
        k = np.abs(np.asarray(eigenvalues.real[0::10]) - t).argmin() * 10  # closest k % 10 == 0 idx
        print('k unspecified, (auto-)choosing:', k)

    eigenvalues = eigenvalues[idx[:k]]
    eigenvectors = eigenvectors[:, idx[:k]]

    correlations = torch.from_numpy(eigenvalues.real).type_as(views[0])
    proj_matrices = torch.split(torch.from_numpy(eigenvectors.real).type_as(views[0]), os)

    return correlations, proj_matrices


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered."""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)
