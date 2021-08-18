import math

import torch
import torch.nn as nn


# class XXDiscriminator32x64(nn.Module):
#     def __init__(self, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         dropout = 0.2
#
#         layers = [
#             # 32x32 -> 16x16
#             sn(nn.Conv2d(1, 32, 4, 2, 1, bias=False)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#
#             # 16x16 -> 8x8
#             sn(nn.Conv2d(32, 64, 4, 2, 1, bias=False)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#
#             # 8x8 -> 4x4
#             sn(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#
#             # 4x4 -> 1x1
#             nn.Conv2d(128, 256, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#
#             nn.Flatten(start_dim=1),
#         ]
#         self.x1_discriminator = nn.Sequential(*layers)
#
#         nf = 64
#         layers = [
#             sn(nn.Conv2d(3, nf, 4, 2, 1, bias=False)),
#             nn.BatchNorm2d(nf),
#             nn.ReLU(True),
#
#             sn(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False)),
#             nn.BatchNorm2d(nf * 2),
#             nn.ReLU(True),
#
#             sn(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False)),
#             nn.BatchNorm2d(nf * 4),
#             nn.ReLU(True),
#
#             sn(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False)),
#             nn.BatchNorm2d(nf * 8),
#             nn.ReLU(True),
#
#             sn(nn.Conv2d(nf * 8, nf * 8, 4, 2, 0, bias=False)),
#             nn.BatchNorm2d(nf * 8),
#             nn.ReLU(True),
#
#             nn.Flatten(start_dim=1)
#         ]
#         self.x2_discriminator = nn.Sequential(*layers)
#
#         layers = [
#             sn(nn.Linear(nf * 8 + 256, 1024)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(1024, 1024)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(1024, output_dim)),
#         ]
#         self.joint_discriminator = nn.Sequential(*layers)
#
#     def forward(self, x1, x2):
#         hx1 = self.x1_discriminator(x1)
#         hx2 = self.x2_discriminator(x2)
#         out = self.joint_discriminator(torch.cat([hx1, hx2], dim=1))
#
#         return out
#
#
# class XXDiscriminator32x64Shared(nn.Module):
#     def __init__(self, x1_discrimination, x2_discrimination, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         dropout = 0.2
#         nf = 64
#
#         self.x1_discrimination = x1_discrimination
#         self.x2_discrimination = x2_discrimination
#         self.discriminator = nn.Sequential(
#             sn(nn.Linear(nf * 8 + 256, 1024)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(1024, 1024)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(1024, output_dim)),
#         )
#
#     def forward(self, x1, x2):
#         hx1 = self.x1_discrimination(x1)
#         hx2 = self.x2_discrimination(x2)
#
#         return self.discriminator(torch.cat([hx1, hx2], dim=1))
#
#
# class XXDiscriminatorConv1dFT(nn.Module):
#     def __init__(self, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         channels = 32
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         use_bias = False
#         self.x1_discrimination = nn.Sequential(
#             # 128 -> 64
#             sn(nn.Conv1d(channels, 64, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#
#             # 64 -> 32
#             sn(nn.Conv1d(64, 128, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#
#             # 32 -> 16
#             sn(nn.Conv1d(128, 128, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#
#             # 16 -> 8
#             sn(nn.Conv1d(128, 256, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#
#             # 8 -> 4
#             sn(nn.Conv1d(256, 256, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#
#             # 4 -> 1
#             sn(nn.Conv1d(256, 256, 4, 1, 0, bias=use_bias)),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#
#             nn.Flatten(start_dim=1),
#         )
#
#         channels = 2048
#         hidden_dim = 256
#         layers = []
#         for i in range(int(math.log2(channels / hidden_dim))):
#             layers += [
#                 sn(nn.Linear(channels // (2 ** i), channels // (2 ** (i + 1)))),
#                 nn.LeakyReLU(inplace=True),
#             ]
#
#         layers += [
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(inplace=True),
#         ]
#         self.x2_discrimination = nn.Sequential(*layers)
#
#         dropout = 0.2
#         self.joint_discriminator = nn.Sequential(
#             sn(nn.Linear(512, 1024)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(1024, 1024)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(1024, output_dim)),
#         )
#
#     def forward(self, x1, x2):
#         x1 = x1.squeeze(dim=1)
#         hx1 = self.x1_discrimination(x1)
#         hx2 = self.x2_discrimination(x2)
#         return self.joint_discriminator(torch.cat([hx1, hx2], dim=1))
#
#
# class XXDiscriminatorConv1d(nn.Module):
#     def __init__(self, channels=32, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         use_bias = False
#         self.x1_discrimination = nn.Sequential(
#             # 128 -> 64
#             sn(nn.Conv1d(channels, 64, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#
#             # 64 -> 32
#             sn(nn.Conv1d(64, 128, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#
#             # 32 -> 16
#             sn(nn.Conv1d(128, 128, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#
#             # 16 -> 8
#             sn(nn.Conv1d(128, 256, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#
#             # 8 -> 4
#             sn(nn.Conv1d(256, 256, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#
#             # 4 -> 1
#             sn(nn.Conv1d(256, 256, 4, 1, 0, bias=use_bias)),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#
#             nn.Flatten(start_dim=1),
#         )
#
#         channels = 3
#         nf = 32
#         self.x2_discrimination = nn.Sequential(
#             # 64 x 64
#             sn(nn.Conv2d(channels, nf, 4, 2, 1)),
#             nn.BatchNorm2d(nf),
#             nn.ReLU(inplace=True),
#
#             # 32 x 32
#             sn(nn.Conv2d(nf, nf * 2, 4, 2, 1)),
#             nn.BatchNorm2d(nf * 2),
#             nn.ReLU(inplace=True),
#
#             # 16 x 16
#             sn(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1)),
#             nn.BatchNorm2d(nf * 4),
#             nn.ReLU(inplace=True),
#
#             # 8 x 8
#             sn(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1)),
#             nn.BatchNorm2d(nf * 8),
#             nn.ReLU(inplace=True),
#
#             # 4 x 4
#             sn(nn.Conv2d(nf * 8, nf * 8, 4, 1, 0)),
#             nn.BatchNorm2d(nf * 8),
#             nn.ReLU(inplace=True),
#
#             nn.Flatten(start_dim=1),
#         )
#
#         dropout = 0.2
#         self.joint_discriminator = nn.Sequential(
#             sn(nn.Linear(512, 1024)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(1024, 1024)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(1024, output_dim)),
#         )
#
#     def forward(self, x1, x2):
#         x1 = x1.squeeze(dim=1)
#         hx1 = self.x1_discrimination(x1)
#         hx2 = self.x2_discrimination(x2)
#         return self.joint_discriminator(torch.cat([hx1, hx2], dim=1))


class XXDiscriminatorFT(nn.Module):
    def __init__(self, emb_size, channels=2048, num_features=32, output_dim=1, spectral_norm=True):
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

        dim_hidden = 256
        layers = []
        for i in range(int(math.log2(channels / dim_hidden))):
            layers += [
                sn(nn.Linear(channels // (2 ** i), channels // (2 ** (i + 1)))),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # layers += [
        #     sn(nn.Linear(dim_hidden, dim_hidden)),
        #     nn.LeakyReLU(0.2, inplace=True),
        # ]
        self.x2_discrimination = nn.Sequential(*layers)

        self.joint_discriminator = nn.Sequential(
            sn(nn.Linear(dim_hidden + num_features * 16, 1024)),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # sn(nn.Linear(1024, 1024)),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, output_dim)),
        )

    def forward(self, x1, x2):
        hx1 = self.x1_discrimination(x1)
        hx2 = self.x2_discrimination(x2)
        return self.joint_discriminator(torch.cat([hx1, hx2], dim=1))

# class XXDiscriminatorFT2(nn.Module):
#     def __init__(self, channels, num_features=32, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         use_bias = False
#         self.x1_discrimination = nn.Sequential(
#             # input size: 1 x 32 x 128
#             sn(nn.Conv2d(1, num_features, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm2d(num_features),
#             nn.ReLU(True),
#             # size: (num_features) x 16 x 64
#
#             sn(nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm2d(num_features * 2),
#             nn.ReLU(True),
#             # size: (num_features * 2) x 8 x 32
#
#             sn(nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=use_bias)),
#             nn.BatchNorm2d(num_features * 4),
#             nn.ReLU(True),
#             # size: (num_features * 4) x 4 x 16
#
#             sn(nn.Conv2d(num_features * 4, num_features * 8, (1, 4), (1, 2), (0, 1), bias=use_bias)),
#             nn.BatchNorm2d(num_features * 8),
#             nn.ReLU(True),
#             # size: (num_features * 8) x 4 x 8
#
#             sn(nn.Conv2d(num_features * 8, num_features * 16, (1, 4), (1, 2), (0, 1), bias=use_bias)),
#             nn.BatchNorm2d(num_features * 16),
#             nn.ReLU(True),
#             # size: (num_features * 8) x 4 x 4
#
#             sn(nn.Conv2d(num_features * 16, num_features * 16, 4, 1, 0, bias=use_bias)),
#             nn.BatchNorm2d(num_features * 16),
#             nn.ReLU(True),
#             # size: (num_features * 16) x 1 x 1
#
#             nn.Flatten(start_dim=1),
#         )
#
#         dim_hidden = 256
#         layers = []
#         for i in range(int(math.log2(channels / dim_hidden))):
#             layers += [
#                 sn(nn.Linear(channels // (2 ** i), channels // (2 ** (i + 1)))),
#                 # nn.ELU(inplace=True)
#                 nn.LeakyReLU(inplace=True),
#             ]
#
#         layers += [
#             sn(nn.Linear(dim_hidden, dim_hidden)),
#             nn.LeakyReLU(inplace=True),
#         ]
#         self.x2_discrimination = nn.Sequential(*layers)
#
#         dropout = 0.2
#         self.joint_discriminator = nn.Sequential(
#             sn(nn.Linear(dim_hidden + num_features * 16, 1024)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(1024, 1024)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(1024, output_dim)),
#         )
#
#     def forward(self, x1, x2):
#         hx1 = self.x1_discrimination(x1)
#         hx2 = self.x2_discrimination(x2)
#         return self.joint_discriminator(torch.cat([hx1, hx2], dim=1))
