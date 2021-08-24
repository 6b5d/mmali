import math

import torch
import torch.nn as nn


class XXDiscriminatorFT(nn.Module):
    def __init__(self, emb_size=128, channels=2048, num_features=32, output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        use_bias = False
        self.x1_discrimination = nn.Sequential(
            sn(nn.Linear(emb_size, 128)),

            # input size: 1 x 32 x 128
            sn(nn.Conv2d(1, num_features, 4, 2, 1, bias=use_bias)),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features) x 16 x 64

            sn(nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=use_bias)),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 2) x 8 x 32

            sn(nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=use_bias)),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 4) x 4 x 16

            sn(nn.Conv2d(num_features * 4, num_features * 8, (1, 4), (1, 2), (0, 1), bias=use_bias)),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 8) x 4 x 8

            sn(nn.Conv2d(num_features * 8, num_features * 16, (1, 4), (1, 2), (0, 1), bias=use_bias)),
            nn.BatchNorm2d(num_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 16) x 4 x 4

            sn(nn.Conv2d(num_features * 16, num_features * 16, 4, 1, 0, bias=use_bias)),
            nn.BatchNorm2d(num_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 16) x 1 x 1

            nn.Flatten(start_dim=1),
        )

        sn = lambda x: x
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
        self.x2_discrimination = nn.Sequential(*layers)

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.joint_discriminator = nn.Sequential(
            sn(nn.Linear(dim_hidden + num_features * 16, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, output_dim)),
        )

    def forward(self, x1, x2):
        hx1 = self.x1_discrimination(x1)
        hx2 = self.x2_discrimination(x2)
        return self.joint_discriminator(torch.cat([hx1, hx2], dim=1))


class XXFTDiscriminator(nn.Module):
    def __init__(self, emb_size=128, channels=2048, num_features=32, output_dim=1, spectral_norm=False):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x

        self.x1_discrimination = nn.Sequential(
            nn.Linear(emb_size, num_features),

            # 32x32 -> 16x16
            sn(nn.Conv2d(1, num_features, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            sn(nn.Conv2d(num_features, num_features * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            sn(nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1
            sn(nn.Conv2d(num_features * 4, 256, 4, 1, 0)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(start_dim=1),
        )

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
        self.x2_discrimination = nn.Sequential(*layers)

        self.joint_discriminator = nn.Sequential(
            sn(nn.Linear(512, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, output_dim)),
        )

    def forward(self, x1, x2):
        hx1 = self.x1_discrimination(x1)
        hx2 = self.x2_discrimination(x2)
        return self.joint_discriminator(torch.cat([hx1, hx2], dim=1))
