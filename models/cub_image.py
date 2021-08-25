import math

import torch
import torch.nn as nn


class EncoderFT(nn.Module):
    def __init__(self, latent_dim, channels=2048):
        super().__init__()

        dim_hidden = 256
        layers = []
        for i in range(int(math.log2(channels / dim_hidden))):
            layers += [
                nn.Linear(channels // (2 ** i), channels // (2 ** (i + 1))),
                nn.ELU(inplace=True)
            ]
        layers += [nn.Linear(dim_hidden, latent_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DecoderFT(nn.Module):
    def __init__(self, latent_dim, channels=2048):
        super().__init__()

        dim_hidden = 256
        layers = []
        for i in range(int(math.log2(channels / dim_hidden))):
            in_dim = latent_dim if i == 0 else dim_hidden * i
            out_dim = dim_hidden if i == 0 else dim_hidden * (2 * i)
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.ELU(inplace=True),
            ]
        layers += [
            nn.Linear(channels // 2, channels),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class XZDiscriminatorFT(nn.Module):
    def __init__(self, latent_dim, channels=2048, output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        dim_hidden = 256
        layers = []
        for i in range(int(math.log2(channels / dim_hidden))):
            layers += [
                sn(nn.Linear(channels // (2 ** i), channels // (2 ** (i + 1)))),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers += [
            sn(nn.Linear(dim_hidden, dim_hidden)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.x_discrimination = nn.Sequential(*layers)

        self.z_discrimination = nn.Sequential(
            sn(nn.Linear(latent_dim, dim_hidden)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(dim_hidden, dim_hidden)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.discriminator = nn.Sequential(
            sn(nn.Linear(2 * dim_hidden, 512)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(512, 512)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(512, output_dim)),
        )

    def forward(self, x, z):
        hx = self.x_discrimination(x)
        hz = self.z_discrimination(z)
        out = self.discriminator(torch.cat([hx, hz], dim=1))
        return out
