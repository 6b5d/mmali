import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim, emb_size=128, num_features=32):
        super().__init__()

        use_bias = False
        self.model = nn.Sequential(
            nn.Linear(emb_size, 128),
            # size: 1 x 32 x 128

            nn.Conv2d(1, num_features, 4, 2, 1, bias=use_bias),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            # size: (num_features) x 16 x 64

            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=use_bias),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(True),
            # size: (num_features * 2) x 8 x 32

            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=use_bias),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(True),
            # size: (num_features * 4) x 4 x 16

            nn.Conv2d(num_features * 4, num_features * 8, (1, 4), (1, 2), (0, 1), bias=use_bias),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(True),
            # size: (num_features * 8) x 4 x 8

            nn.Conv2d(num_features * 8, num_features * 16, (1, 4), (1, 2), (0, 1), bias=use_bias),
            nn.BatchNorm2d(num_features * 16),
            nn.ReLU(True),
            # size: (num_features * 16) x 4 x 4

            nn.Conv2d(num_features * 16, latent_dim, 4, 1, 0, bias=True),
            # size: (num_features * 16) x 1 x 1
            nn.Flatten(start_dim=1)
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, emb_size=128, num_features=32):
        super().__init__()

        use_bias = False
        self.model = nn.Sequential(
            nn.Unflatten(1, (latent_dim, 1, 1)),

            nn.ConvTranspose2d(latent_dim, num_features * 16, 4, 1, 0, bias=use_bias),
            nn.BatchNorm2d(num_features * 16),
            nn.ReLU(True),
            # size: (num_features * 16) x 4 x 4

            nn.ConvTranspose2d(num_features * 16, num_features * 8, (1, 4), (1, 2), (0, 1), bias=use_bias),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(True),
            # size: (num_features * 8) x 4 x 8

            nn.ConvTranspose2d(num_features * 8, num_features * 4, (1, 4), (1, 2), (0, 1), bias=use_bias),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(True),
            # size: (num_features * 4) x 8 x 32

            nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1, bias=use_bias),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(True),
            # size: (num_features * 2) x 16 x 64

            nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1, bias=use_bias),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            # size: (num_features) x 32 x 128

            nn.ConvTranspose2d(num_features, 1, 4, 2, 1, bias=True),
            nn.ReLU(True),

            nn.Linear(128, emb_size),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.model(z)


class XZDiscriminator(nn.Module):
    def __init__(self, latent_dim, emb_size=128, num_features=32, output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        use_bias = True
        self.x_discrimination = nn.Sequential(
            sn(nn.Linear(emb_size, 128)),
            # size: 1 x 32 x 128

            sn(nn.Conv2d(1, num_features, 4, 2, 1, bias=use_bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features) x 16 x 64

            sn(nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=use_bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 2) x 8 x 32

            sn(nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=use_bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 4) x 4 x 16

            sn(nn.Conv2d(num_features * 4, num_features * 8, (1, 4), (1, 2), (0, 1), bias=use_bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 8) x 4 x 8

            sn(nn.Conv2d(num_features * 8, num_features * 16, (1, 4), (1, 2), (0, 1), bias=use_bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 8) x 4 x 4

            sn(nn.Conv2d(num_features * 16, num_features * 16, 4, 1, 0, bias=use_bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (num_features * 16) x 1 x 1

            nn.Flatten(start_dim=1),
        )

        self.z_discrimination = nn.Sequential(
            sn(nn.Linear(latent_dim, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            sn(nn.Linear(512, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )

        self.discriminator = nn.Sequential(
            sn(nn.Linear(1024, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            sn(nn.Linear(1024, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            sn(nn.Linear(1024, output_dim)),
        )

    def forward(self, x, z):
        hx = self.x_discrimination(x)
        hz = self.z_discrimination(z)
        out = self.discriminator(torch.cat([hx, hz], dim=1))
        return out

# class Encoder32x32(nn.Module):
#     def __init__(self, latent_dim, channels=1):
#         super().__init__()
#
#         use_bias = True
#         self.model = nn.Sequential(
#             # 32x32 -> 16x16
#             nn.Conv2d(channels, 32, 4, 2, 1, bias=use_bias),
#             nn.ReLU(inplace=True),
#
#             # 16x16 -> 8x8
#             nn.Conv2d(32, 64, 4, 2, 1, bias=use_bias),
#             nn.ReLU(inplace=True),
#
#             # 8x8 -> 4x4
#             nn.Conv2d(64, 128, 4, 2, 1, bias=use_bias),
#             nn.ReLU(inplace=True),
#
#             # 4x4 -> 1x1
#             nn.Conv2d(128, latent_dim, 4, 1, 0),
#             nn.Flatten(start_dim=1),
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
#
# class Decoder32x32(nn.Module):
#     def __init__(self, latent_dim, channels=1):
#         super().__init__()
#
#         self.model = nn.Sequential(
#             nn.Unflatten(1, (latent_dim, 1, 1)),
#
#             # 1x1 -> 4x4
#             nn.ConvTranspose2d(latent_dim, 128, 4, 1, 0),
#             nn.ReLU(inplace=True),
#
#             # 4x4 -> 8x8
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             # 8x8 -> 16x16
#             nn.ConvTranspose2d(64, 32, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             # 16x16 -> 32x32
#             nn.ConvTranspose2d(32, channels, 4, 2, 1),
#             nn.Sigmoid(),
#             # nn.Tanh(),
#         )
#
#     def forward(self, z):
#         return self.model(z)
#
#
# class XZDiscriminator32x32(nn.Module):
#     def __init__(self, latent_dim, channels=1, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         dropout = 0.2
#
#         self.x_discrimination = nn.Sequential(
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
#         self.z_discrimination = nn.Sequential(
#             sn(nn.Linear(latent_dim, 256)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(256, 256)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#         )
#
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
#     def forward(self, x, z):
#         hx = self.x_discrimination(x)
#         hz = self.z_discrimination(z)
#         return self.joint_discriminator(torch.cat([hx, hz], dim=1))
#
#
# class XZDiscriminator32x32Shared(nn.Module):
#     def __init__(self, x_discrimination=None, z_discrimination=None,
#                  latent_dim=None, channels=None,
#                  output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         dropout = 0.2
#
#         if x_discrimination is None:
#             x_discrimination = XDiscrimination32x32(channels=channels, spectral_norm=spectral_norm)
#         if z_discrimination is None:
#             z_discrimination = ZDiscrimination32x32(latent_dim=latent_dim, spectral_norm=spectral_norm)
#
#         self.x_discrimination = x_discrimination
#         self.z_discrimination = z_discrimination
#
#         self.discriminator = nn.Sequential(
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
#     def forward(self, x, z):
#         hx = self.x_discrimination(x)
#         hz = self.z_discrimination(z)
#         return self.discriminator(torch.cat([hx, hz], dim=1))


# input 1x32x32
# output: 256
# class XDiscrimination32x32(nn.Module):
#     def __init__(self, channels=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         self.x_discrimination = nn.Sequential(
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
#     def forward(self, x):
#         return self.x_discrimination(x)


# input: latent_dim
# output: 256
# class ZDiscrimination32x32(nn.Module):
#     def __init__(self, latent_dim, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         dropout = 0.2
#
#         self.z_discrimination = nn.Sequential(
#             sn(nn.Linear(latent_dim, 256)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(256, 256)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#         )
#
#     def forward(self, z):
#         return self.z_discrimination(z)


# class StyleContentEncoder32x32(nn.Module):
#     def __init__(self, style_dim, content_dim, channels=1):
#         super().__init__()
#
#         self.encoder = nn.Sequential(
#             # 32x32 -> 16x16
#             nn.Conv2d(channels, 32, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             # 16x16 -> 8x8
#             nn.Conv2d(32, 64, 4, 2, 1),
#             nn.ReLU(inplace=True),
#         )
#         self.style_encoder = nn.Sequential(
#             # 8x8 -> 4x4
#             nn.Conv2d(64, 128, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             # 4x4 -> 1x1
#             nn.Conv2d(128, style_dim, 4, 1, 0),
#             nn.Flatten(start_dim=1),
#         )
#         self.content_encoder = nn.Sequential(
#             # 8x8 -> 4x4
#             nn.Conv2d(64, 128, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             # 4x4 -> 1x1
#             nn.Conv2d(128, content_dim, 4, 1, 0),
#             nn.Flatten(start_dim=1),
#         )
#
#     def forward(self, x):
#         h = self.encoder(x)
#         s = self.style_encoder(h)
#         c = self.content_encoder(h)
#         return s, c


# class EncoderConv1d(nn.Module):
#     def __init__(self, latent_dim, channels=32):
#         super().__init__()
#
#         use_bias = False
#         self.model = nn.Sequential(
#             # channels (=32) * 128 -> 64 * 64
#             nn.Conv1d(channels, 64, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#
#             # 64 * 64 -> 128 * 32
#             nn.Conv1d(64, 128, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#
#             # 128 * 32 -> 128 * 16
#             nn.Conv1d(128, 128, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#
#             # 128 * 16 -> 256 * 8
#             nn.Conv1d(128, 256, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#
#             # 256 * 8 -> 256 * 4
#             nn.Conv1d(256, 256, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#
#             # 256 * 4 -> latent_dim * 1
#             nn.Conv1d(256, latent_dim, 4, 1, 0),
#
#             nn.Flatten(start_dim=1),
#         )
#
#     def forward(self, x):
#         x = x.squeeze(dim=1)
#         return self.model(x)
#
#
# class DecoderConv1d(nn.Module):
#     def __init__(self, latent_dim, channels=32):
#         super().__init__()
#         use_bias = False
#         self.model = nn.Sequential(
#             nn.Unflatten(1, (latent_dim, 1)),
#
#             # latent_dim * 1 -> 256 * 4
#             nn.ConvTranspose1d(latent_dim, 256, 4, 1, 0, bias=use_bias),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#
#             # 256 * 4 -> 256 * 8
#             nn.ConvTranspose1d(256, 256, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#
#             # 256 * 8 -> 128 * 16
#             nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#
#             # 128 * 16 -> 128 * 32
#             nn.ConvTranspose1d(128, 128, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#
#             # 128 * 32 -> 64 * 64
#             nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=use_bias),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#
#             # 64 * 64 -> channels (=32) * 128
#             nn.ConvTranspose1d(64, channels, 4, 2, 1),
#
#             # nn.Sigmoid(),
#         )
#
#     def forward(self, z):
#         x = self.model(z)
#
#         x = x.unsqueeze(dim=1)
#         return x
#
#
# class XZDiscriminatorConv1d(nn.Module):
#     def __init__(self, latent_dim, channels=32, output_dim=1, spectral_norm=True):
#         super().__init__()
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#
#         # self.x_discrimination = nn.Sequential(
#         self.discriminator = nn.Sequential(
#             # 128 -> 64
#             sn(nn.Conv1d(channels + 1, 64, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 64 -> 32
#             sn(nn.Conv1d(64, 128, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 32 -> 16
#             sn(nn.Conv1d(128, 128, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 16 -> 8
#             sn(nn.Conv1d(128, 256, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 8 -> 4
#             sn(nn.Conv1d(256, 256, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 4 -> 1
#             sn(nn.Conv1d(256, output_dim, 4, 1, 0)),
#             # nn.ReLU(inplace=True),
#
#             nn.Flatten(start_dim=1),
#         )
#
#         # dropout = 0.2
#         # self.z_discrimination = nn.Sequential(
#         #     sn(nn.Linear(latent_dim, 256)),
#         #     nn.Dropout(dropout),
#         #     nn.LeakyReLU(inplace=True),
#         #
#         #     sn(nn.Linear(256, 256)),
#         #     nn.Dropout(dropout),
#         #     nn.LeakyReLU(inplace=True),
#         # )
#
#         # self.joint_discriminator = nn.Sequential(
#         #     sn(nn.Linear(512, 1024)),
#         #     nn.Dropout(dropout),
#         #     nn.LeakyReLU(inplace=True),
#         #
#         #     sn(nn.Linear(1024, 1024)),
#         #     nn.Dropout(dropout),
#         #     nn.LeakyReLU(inplace=True),
#         #
#         #     sn(nn.Linear(1024, output_dim)),
#         # )
#
#     def forward(self, x, z):
#         x = x.squeeze(dim=1)
#         # hx = self.x_discrimination(x)
#         # hz = self.z_discrimination(z)
#         # return self.joint_discriminator(torch.cat([hx, hz], dim=1))
#         z = z.unsqueeze(dim=1)
#         return self.discriminator(torch.cat([x, z], dim=1))
#
#
# class EncoderConv1d2(nn.Module):
#     def __init__(self, latent_dim, channels=128):
#         super().__init__()
#
#         self.model = nn.Sequential(
#             nn.Conv1d(channels, 128, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             nn.Conv1d(128, 256, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             nn.Conv1d(256, 256, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             nn.Conv1d(256, latent_dim, 4, 1, 0),
#
#             nn.Flatten(start_dim=1),
#         )
#
#     def forward(self, x):
#         # x: N, 1, 32, 128
#         x = x.squeeze(dim=1).permute(0, 2, 1)
#         return self.model(x)
#
#
# class DecoderConv1d2(nn.Module):
#     def __init__(self, latent_dim, channels=128):
#         super().__init__()
#
#         self.model = nn.Sequential(
#             nn.Unflatten(1, (latent_dim, 1)),
#
#             nn.ConvTranspose1d(latent_dim, 256, 4, 1, 0),
#             nn.ReLU(inplace=True),
#
#             nn.ConvTranspose1d(256, 256, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             nn.ConvTranspose1d(256, 128, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             nn.ConvTranspose1d(128, channels, 4, 2, 1),
#         )
#
#     def forward(self, z):
#         x = self.model(z)
#         x = x.permute(0, 2, 1).unsqueeze(dim=1)
#         return x
#
#
# class XZDiscriminatorConv1d2(nn.Module):
#     def __init__(self, latent_dim, channels=128, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         self.x_discrimination = nn.Sequential(
#             nn.Conv1d(channels, 128, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             nn.Conv1d(128, 256, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             nn.Conv1d(256, 256, 4, 2, 1),
#             nn.ReLU(inplace=True),
#
#             nn.Conv1d(256, 256, 4, 1, 0),
#
#             nn.Flatten(start_dim=1),
#         )
#
#         dropout = 0.2
#         self.z_discrimination = nn.Sequential(
#             sn(nn.Linear(latent_dim, 256)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(256, 256)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(inplace=True),
#         )
#
#         self.discriminator = nn.Sequential(
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
#     def forward(self, x, z):
#         x = x.squeeze(dim=1).permute(0, 2, 1)
#         hx = self.x_discrimination(x)
#         hz = self.z_discrimination(z)
#         out = self.discriminator(torch.cat([hx, hz], dim=1))
#         return out
