import math

import torch
import torch.nn as nn


class XXDiscriminatorFT(nn.Module):
    def __init__(self, channels=2048, num_features=32, output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        use_bias = False
        self.x1_discrimination = nn.Sequential(
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


class XXDiscriminatorDot(nn.Module):
    def __init__(self, x1_discrimination, x2_discrimination, hidden_dim=256, spectral_norm=True):
        super().__init__()

        self.x1_discrimination = x1_discrimination
        self.x2_discrimination = x2_discrimination
        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.fc1 = sn(nn.Linear(512, hidden_dim))
        self.fc2 = sn(nn.Linear(256, hidden_dim))

    def forward(self, x1, x2):
        hx1 = self.fc1(self.x1_discrimination(x1))
        hx2 = self.fc2(self.x2_discrimination(x2))
        out = torch.sum(hx1 * hx2, dim=1, keepdim=True)
        return out


class XXDiscriminatorDot64(nn.Module):
    def __init__(self, x1_discrimination, x2_discrimination, hidden_dim=256, spectral_norm=True):
        super().__init__()

        self.x1_discrimination = x1_discrimination
        self.x2_discrimination = x2_discrimination
        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.fc1 = sn(nn.Linear(256, hidden_dim))
        self.fc2 = sn(nn.Linear(256, hidden_dim))

    def forward(self, x1, x2):
        hx1 = self.fc1(self.x1_discrimination(x1))
        hx2 = self.fc2(self.x2_discrimination(x2))
        out = torch.sum(hx1 * hx2, dim=1, keepdim=True)
        return out


class XXDiscriminatorFTDot64(nn.Module):
    def __init__(self, channels=2048, num_features=32, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        use_bias = True
        self.x1_discrimination = nn.Sequential(
            # size: 1 x 32 x 64

            sn(nn.Conv2d(1, num_features, 4, 2, 1, bias=use_bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features) x 16 x 32

            sn(nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=use_bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 2) x 8 x 16

            sn(nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=use_bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 4) x 4 x 8

            sn(nn.Conv2d(num_features * 4, num_features * 8, (1, 4), (1, 2), (0, 1), bias=use_bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 8) x 4 x 4

            sn(nn.Conv2d(num_features * 8, num_features * 8, 4, 1, 0, bias=use_bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 16) x 1 x 1

            nn.Flatten(start_dim=1),
        )

        self.x2_discrimination = nn.Sequential(
            sn(nn.Linear(channels, channels // 2)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels // 2, channels // 4)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels // 4, channels // 8)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels // 8, channels // 8)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc1 = sn(nn.Linear(256, 256))
        self.fc2 = sn(nn.Linear(256, 256))

    def forward(self, x1, x2):
        hx1 = self.fc1(self.x1_discrimination(x1))
        hx2 = self.fc2(self.x2_discrimination(x2))
        out = torch.sum(hx1 * hx2, dim=1, keepdim=True)
        return out


class FTXXDiscriminatorDot64(nn.Module):
    def __init__(self, x1_discrimination, x2_discrimination, hidden_dim=128, spectral_norm=True):
        super().__init__()

        self.x1_discrimination = x1_discrimination
        self.x2_discrimination = x2_discrimination
        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.fc1 = sn(nn.Linear(128, hidden_dim))
        self.fc2 = sn(nn.Linear(256, hidden_dim))

    def forward(self, x1, x2):
        hx1 = self.fc1(self.x1_discrimination(x1))
        hx2 = self.fc2(self.x2_discrimination(x2))
        out = torch.sum(hx1 * hx2, dim=1, keepdim=True)
        return out


class DotDiscriminatorFT(nn.Module):
    def __init__(self, channels1=1024, channels2=2048, hidden_dim=256, spectral_norm=True):
        super().__init__()
        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.x1_discrimination = nn.Sequential(
            sn(nn.Linear(channels1, channels1 // 2)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels1 // 2, channels1 // 4)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels1 // 4, channels1 // 8)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels1 // 8, channels1 // 8)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.x2_discrimination = nn.Sequential(
            sn(nn.Linear(channels2, channels2 // 2)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels2 // 2, channels2 // 4)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels2 // 4, channels2 // 8)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels2 // 8, channels2 // 8)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc1 = sn(nn.Linear(128, hidden_dim))
        self.fc2 = sn(nn.Linear(256, hidden_dim))

    def forward(self, x1, x2):
        hx1 = self.fc1(self.x1_discrimination(x1))
        hx2 = self.fc2(self.x2_discrimination(x2))
        out = torch.sum(hx1 * hx2, dim=1, keepdim=True)
        return out


class XXDiscriminatorFTFT(nn.Module):
    def __init__(self, channels1=1024, channels2=2048, output_dim=1, spectral_norm=True):
        super().__init__()
        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x

        self.x1_discrimination = nn.Sequential(
            sn(nn.Linear(channels1, channels1 // 2)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels1 // 2, channels1 // 4)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels1 // 4, channels1 // 8)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels1 // 8, channels1 // 8)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.x2_discrimination = nn.Sequential(
            sn(nn.Linear(channels2, channels2 // 2)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels2 // 2, channels2 // 4)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels2 // 4, channels2 // 8)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels2 // 8, channels2 // 8)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.joint_discriminator = nn.Sequential(
            sn(nn.Linear(128 + 256, 512)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(512, 512)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(512, output_dim)),
        )

    def forward(self, x1, x2):
        hx1 = self.x1_discrimination(x1)
        hx2 = self.x2_discrimination(x2)
        return self.joint_discriminator(torch.cat([hx1, hx2], dim=1))
