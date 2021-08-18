import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class XXDiscriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), channels=3, num_features=32, output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        layers = [
            nn.Flatten(start_dim=1),

            sn(nn.Linear(int(np.prod(img_shape)), 256)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.x1_discrimination = nn.Sequential(*layers)

        self.x2_discrimination = nn.Sequential(
            # 32x32 -> 16x16
            sn(nn.Conv2d(channels, num_features, 4, 2, 1)),
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

        self.joint_discriminator = nn.Sequential(
            sn(nn.Linear(512, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, output_dim)),
        )

    def forward(self, x1, x2):
        hx1 = self.x1_discrimination(x1)
        hx2 = self.x2_discrimination(x2)
        out = self.joint_discriminator(torch.cat([hx1, hx2], dim=1))
        return out


# class XXDiscriminator2(nn.Module):
#     def __init__(self, img_shape=(1, 28, 28), channels=3, hidden_dim=400, extra_layers=0, output_dim=1,
#                  spectral_norm=True):
#         super().__init__()
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         dropout = 0.2
#         layers = [
#             nn.Flatten(start_dim=1),
#
#             sn(nn.Linear(int(np.prod(img_shape)), hidden_dim)),
#             nn.LeakyReLU(inplace=True),
#         ]
#
#         for i in range(extra_layers):
#             layers += [
#                 sn(nn.Linear(hidden_dim, hidden_dim)),
#                 nn.LeakyReLU(inplace=True),
#             ]
#
#         layers += [
#             sn(nn.Linear(hidden_dim, 256)),
#             nn.LeakyReLU(inplace=True),
#         ]
#         self.x1_discrimination = nn.Sequential(*layers)
#
#         self.x2_discrimination = nn.Sequential(
#             # 32x32 -> 16x16
#             sn(nn.Conv2d(channels, 32, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 16x16 -> 8x8
#             sn(nn.Conv2d(32, 64, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 8x8 -> 4x4
#             sn(nn.Conv2d(64, 128, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 4x4 -> 1x1
#             sn(nn.Conv2d(128, 256, 4, 1, 0)),
#             nn.ReLU(inplace=True),
#
#             nn.Flatten(start_dim=1),
#         )
#
#         self.joint_discriminator = nn.Sequential(
#             sn(nn.Linear(512, 1024)),
#             # nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(1024, 1024)),
#             # nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(1024, output_dim)),
#         )
#
#     def forward(self, x1, x2):
#         hx1 = self.x1_discrimination(x1)
#         hx2 = self.x2_discrimination(x2)
#         out = self.joint_discriminator(torch.cat([hx1, hx2], dim=1))
#         return out


class XXDiscriminatorConv(nn.Module):
    def __init__(self, channels=4, output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        use_bias = True
        self.model = nn.Sequential(
            # 32x32 -> 16x16
            sn(nn.Conv2d(channels, 32, 4, 2, 1, bias=use_bias)),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 16x16 -> 8x8
            sn(nn.Conv2d(32, 64, 4, 2, 1, bias=use_bias)),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 8x8 -> 4x4
            sn(nn.Conv2d(64, 128, 4, 2, 1, bias=use_bias)),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 4x4 -> 1x1
            sn(nn.Conv2d(128, 256, 4, 1, 0, bias=use_bias)),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            sn(nn.Conv2d(256, output_dim, 1, 1, 0)),
            nn.Flatten(start_dim=1),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, (32, 32))
        return self.model(torch.cat([x1, x2], dim=1))


class XXFeatureDiscriminator(nn.Module):
    def __init__(self, x1_feature, x2_feature, hidden_dim=512, output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.x1_feature = x1_feature
        self.x2_feature = x2_feature
        self.joint_discriminator = nn.Sequential(
            sn(nn.Linear(hidden_dim, 512)),
            nn.LeakyReLU(0.2, inplace=True),

            # sn(nn.Linear(1024, 1024)),
            # nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(512, output_dim)),
        )

    def forward(self, x1, x2):
        hx1 = self.x1_feature(x1)
        hx2 = self.x2_feature(x2)
        return self.joint_discriminator(torch.cat([hx1, hx2], dim=1))
#
#
# class XXZDiscriminator(nn.Module):
#     def __init__(self, latent_dim,
#                  img_shape=(1, 28, 28), channels=3, hidden_dim=400, extra_layers=0, output_dim=1, spectral_norm=True):
#         super().__init__()
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#
#         layers = [
#             nn.Flatten(start_dim=1),
#
#             sn(nn.Linear(int(np.prod(img_shape)), hidden_dim)),
#             nn.ReLU(inplace=True),
#         ]
#
#         for i in range(extra_layers):
#             layers += [
#                 sn(nn.Linear(hidden_dim, hidden_dim)),
#                 nn.ReLU(inplace=True),
#             ]
#
#         layers += [
#             sn(nn.Linear(hidden_dim, 256)),
#             nn.ReLU(inplace=True),
#         ]
#         self.x1_discriminator = nn.Sequential(*layers)
#
#         layers = [
#             sn(nn.Conv2d(channels, 32, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             sn(nn.Conv2d(32, 64, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             sn(nn.Conv2d(64, 128, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             sn(nn.Conv2d(128, 256, 4, 1, 0)),
#             nn.ReLU(inplace=True),
#
#             nn.Flatten(start_dim=1),
#         ]
#         self.x2_discriminator = nn.Sequential(*layers)
#
#         layers = [
#             sn(nn.Linear(latent_dim, 256)),
#             nn.ReLU(inplace=True),
#
#             sn(nn.Linear(256, 256)),
#             nn.ReLU(inplace=True),
#         ]
#         self.z_discriminator = nn.Sequential(*layers)
#
#         layers = [
#             sn(nn.Linear(256 + 256 + 256, 512)),
#             nn.ReLU(inplace=True),
#
#             sn(nn.Linear(512, 512)),
#             nn.ReLU(inplace=True),
#
#             sn(nn.Linear(512, output_dim)),
#         ]
#         self.joint_discriminator = nn.Sequential(*layers)
#
#     def forward(self, x1, x2, z):
#         hx1 = self.x1_discriminator(x1)
#         hx2 = self.x2_discriminator(x2)
#         hz = self.z_discriminator(z)
#
#         out = self.joint_discriminator(torch.cat([hx1, hx2, hz], dim=1))
#         return out
