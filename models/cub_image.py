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
            nn.Sigmoid(),
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

        # layers += [
        #     nn.Linear(dim_hidden, dim_hidden),
        #     nn.LeakyReLU(0.2, inplace=True),
        # ]
        self.x_discrimination = nn.Sequential(*layers)

        self.z_discrimination = nn.Sequential(
            sn(nn.Linear(latent_dim, dim_hidden)),
            nn.LeakyReLU(0.2, inplace=True),

            # sn(nn.Linear(dim_hidden, dim_hidden)),
            # nn.LeakyReLU(0.2, inplace=True),
        )

        self.discriminator = nn.Sequential(
            sn(nn.Linear(2 * dim_hidden, 512)),
            nn.LeakyReLU(0.2, inplace=True),

            # sn(nn.Linear(1024, 1024)),
            # nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(512, output_dim)),
        )

    def forward(self, x, z):
        hx = self.x_discrimination(x)
        hz = self.z_discrimination(z)
        out = self.discriminator(torch.cat([hx, hz], dim=1))
        return out

# class XZDiscriminatorFT2(nn.Module):
#     def __init__(self, latent_dim, channels=2048, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         self.discriminator = nn.Sequential(
#             sn(nn.Linear(latent_dim + channels, 512)),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             sn(nn.Linear(512, 512)),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             sn(nn.Linear(512, 512)),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             sn(nn.Linear(512, output_dim)),
#         )
#
#     def forward(self, x, z):
#         return self.discriminator(torch.cat([x, z], dim=1))
#
#
# class Encoder64(nn.Module):
#     def __init__(self, latent_dim, channels=3):
#         super().__init__()
#         nf = 32
#         use_bias = False
#         self.model = nn.Sequential(
#             # 64x64 -> 32x32
#             nn.Conv2d(channels, nf, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm2d(nf),
#             nn.ReLU(True),
#
#             # 32x32 -> 16x16
#             nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm2d(nf * 2),
#             nn.ReLU(True),
#
#             # 16x16 -> 8x8
#             nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm2d(nf * 4),
#             nn.ReLU(True),
#
#             # 8x8 -> 4x4
#             nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm2d(nf * 8),
#             nn.ReLU(True),
#
#             # 4x4 -> 1x1
#             nn.Conv2d(nf * 8, latent_dim, 4, 2, 0),
#             nn.Flatten(start_dim=1),
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
#
# class Decoder64(nn.Module):
#     def __init__(self, latent_dim, channels=3):
#         super().__init__()
#         nf = 32
#         use_bias = False
#         self.model = nn.Sequential(
#             nn.Unflatten(1, (latent_dim, 1, 1)),
#
#             # 1x1
#             nn.ConvTranspose2d(latent_dim, nf * 8, 4, 1, 0, bias=use_bias),
#             nn.BatchNorm2d(nf * 8),
#             nn.ReLU(True),
#
#             # 4x4
#             nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm2d(nf * 4),
#             nn.ReLU(True),
#
#             # 8x8
#             nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm2d(nf * 2),
#             nn.ReLU(True),
#
#             # 16x16
#             nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm2d(nf),
#             nn.ReLU(True),
#
#             # 32x32
#             nn.ConvTranspose2d(nf, channels, 4, 2, 1),
#             nn.Sigmoid()
#             # 64x64
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
#
# class XZDiscriminator64(nn.Module):
#     def __init__(self, latent_dim, channels=3, output_dim=1, spectral_norm=True):
#         super().__init__()
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         nf = 32
#         dropout = 0.2
#
#         self.x_discrimination = nn.Sequential(
#             # 64 x 64
#             sn(nn.Conv2d(channels, nf, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 32 x 32
#             sn(nn.Conv2d(nf, nf * 2, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 16 x 16
#             sn(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 8 x 8
#             sn(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 4 x 4
#             sn(nn.Conv2d(nf * 8, nf * 8, 4, 1, 0)),
#             nn.ReLU(inplace=True),
#
#             nn.Flatten(start_dim=1),
#         )
#
#         self.z_discriminator = nn.Sequential(
#             sn(nn.Linear(latent_dim, nf * 8)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(nf * 8, nf * 8)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#         )
#
#         self.discriminator = nn.Sequential(
#             sn(nn.Linear(nf * 16, nf * 16)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(nf * 16, nf * 16)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(nf * 16, output_dim)),
#         )
#
#     def forward(self, x, z):
#         hx = self.x_discrimination(x)
#         hz = self.z_discriminator(z)
#         return self.discriminator(torch.cat([hx, hz], dim=1))


# class XZDiscriminator64FM(nn.Module):
#     def __init__(self, latent_dim, channels=3, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         nf = 64
#         dropout = 0.2
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#
#         self.x_layer1 = nn.Sequential(
#             # 64x64
#             sn(nn.Conv2d(channels, nf, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#         )
#         self.x_layer2 = nn.Sequential(
#             # 32x32
#             sn(nn.Conv2d(nf, nf * 2, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#         )
#         self.x_layer3 = nn.Sequential(
#             # 16x16
#             sn(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#         )
#         self.x_layer4 = nn.Sequential(
#             # 8x8
#             sn(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#         )
#         self.x_layer5 = nn.Sequential(
#             # 4x4
#             sn(nn.Conv2d(nf * 8, nf * 8, 4, 1, 0)),
#             nn.ReLU(inplace=True),
#             nn.Flatten(start_dim=1),
#         )
#
#         self.z_layer1 = nn.Sequential(
#             sn(nn.Linear(latent_dim, nf * 8)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#         )
#         self.z_layer2 = nn.Sequential(
#             sn(nn.Linear(nf * 8, nf * 8)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#         )
#
#         self.joint_layer1 = nn.Sequential(
#             sn(nn.Linear(nf * 16, nf * 16)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#         )
#         self.joint_layer2 = nn.Sequential(
#             sn(nn.Linear(nf * 16, nf * 16)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#         )
#         self.joint_layer3 = sn(nn.Linear(nf * 16, output_dim))
#
#     def forward(self, x, z, feature=False):
#         hx = x
#         x_feats = []
#         for i in [1, 2, 3, 4, 5]:
#             hx = getattr(self, 'x_layer{}'.format(i))(hx)
#             x_feats.append(hx)
#
#         hz = z
#         z_feats = []
#         for i in [1, 2]:
#             hz = getattr(self, 'z_layer{}'.format(i))(hz)
#             z_feats.append(hz)
#
#         h = torch.cat([hx, hz], dim=1)
#         joint_feats = []
#         for i in [1, 2, 3]:
#             h = getattr(self, 'joint_layer{}'.format(i))(h)
#             joint_feats.append(h)
#
#         if feature:
#             return h, x_feats[2:]
#
#         return h


# class XZDiscriminator64Shared(nn.Module):
#     def __init__(self, x_discrimination=None, z_discrimination=None,
#                  latent_dim=None, channels=None,
#                  output_dim=1, spectral_norm=True):
#         super().__init__()
#         nf = 64
#         dropout = 0.2
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#
#         if x_discrimination is None:
#             x_discrimination = XDiscrimination64(channels=channels, spectral_norm=spectral_norm)
#         if z_discrimination is None:
#             z_discrimination = ZDiscrimination64(latent_dim=latent_dim, spectral_norm=spectral_norm)
#
#         self.x_discrimination = x_discrimination
#         self.z_discrimination = z_discrimination
#
#         layers = [
#             sn(nn.Linear(nf * 16, nf * 16)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(nf * 16, nf * 16)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(nf * 16, output_dim)),
#         ]
#         self.discriminator = nn.Sequential(*layers)
#
#     def forward(self, x, z):
#         hx = self.x_discrimination(x)
#         hz = self.z_discrimination(z)
#         return self.discriminator(torch.cat([hx, hz], dim=1))


# input: 3x64x64
# output: nf * 8
# class XDiscrimination64(nn.Module):
#     def __init__(self, channels=3, spectral_norm=True):
#         super().__init__()
#         nf = 64
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         self.x_discrimination = nn.Sequential(
#             # 64 x 64
#             sn(nn.Conv2d(channels, nf, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 32 x 32
#             sn(nn.Conv2d(nf, nf * 2, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 16 x 16
#             sn(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             #  8 x 8
#             sn(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 4 x 4
#             sn(nn.Conv2d(nf * 8, nf * 8, 4, 1, 0)),
#             nn.ReLU(inplace=True),
#
#             nn.Flatten(start_dim=1),
#         )
#
#     def forward(self, x):
#         return self.x_discrimination(x)


# input: latent_dim
# output: nf * 8
# class ZDiscrimination64(nn.Module):
#     def __init__(self, latent_dim, spectral_norm=True):
#         super().__init__()
#
#         nf = 64
#         dropout = 0.2
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#
#         self.z_discrimination = nn.Sequential(
#             sn(nn.Linear(latent_dim, nf * 8)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(nf * 8, nf * 8)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#         )
#
#     def forward(self, z):
#         return self.z_discrimination(z)
