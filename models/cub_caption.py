import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim, num_features=32):
        super().__init__()

        use_bias = False
        self.model = nn.Sequential(
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
    def __init__(self, latent_dim, num_features=32):
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
            # nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class XZDiscriminator(nn.Module):
    def __init__(self, latent_dim, num_features=32, output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        use_bias = True
        self.x_discrimination = nn.Sequential(
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

            sn(nn.Conv2d(num_features * 16, 256, 1, 1, 0, bias=use_bias)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(start_dim=1),
        )

        self.z_discrimination = nn.Sequential(
            sn(nn.Linear(latent_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(256, 256)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.discriminator = nn.Sequential(
            sn(nn.Linear(512, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, output_dim)),
        )

    def forward(self, x, z):
        hx = self.x_discrimination(x)
        hz = self.z_discrimination(z)
        out = self.discriminator(torch.cat([hx, hz], dim=1))
        return out


class Encoder64(nn.Module):
    def __init__(self, latent_dim, num_features=32):
        super().__init__()

        use_bias = False
        self.model = nn.Sequential(
            # size: 1 x 32 x 64

            nn.Conv2d(1, num_features, 4, 2, 1, bias=use_bias),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            # size: (num_features) x 16 x 32

            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=use_bias),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(True),
            # size: (num_features * 2) x 8 x 16

            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=use_bias),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(True),
            # size: (num_features * 4) x 4 x 8

            nn.Conv2d(num_features * 4, num_features * 8, (1, 4), (1, 2), (0, 1), bias=use_bias),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(True),
            # size: (num_features * 16) x 4 x 4

            nn.Conv2d(num_features * 8, latent_dim, 4, 1, 0, bias=True),
            # size: (num_features * 16) x 1 x 1
            nn.Flatten(start_dim=1)
        )

    def forward(self, x):
        return self.model(x)


class Decoder64(nn.Module):
    def __init__(self, latent_dim, num_features=32):
        super().__init__()

        use_bias = False
        self.model = nn.Sequential(
            nn.Unflatten(1, (latent_dim, 1, 1)),

            nn.ConvTranspose2d(latent_dim, num_features * 8, 4, 1, 0, bias=use_bias),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(True),
            # size: (num_features * 16) x 4 x 4

            nn.ConvTranspose2d(num_features * 8, num_features * 4, (1, 4), (1, 2), (0, 1), bias=use_bias),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(True),
            # size: (num_features * 8) x 4 x 8

            nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1, bias=use_bias),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(True),
            # size: (num_features * 2) x 16 x 32

            nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1, bias=use_bias),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            # size: (num_features) x 32 x 64

            nn.ConvTranspose2d(num_features, 1, 4, 2, 1, bias=True),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class XZDiscriminator64(nn.Module):
    def __init__(self, latent_dim, num_features=32, output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        use_bias = True
        self.x_discrimination = nn.Sequential(
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

        self.z_discrimination = nn.Sequential(
            sn(nn.Linear(latent_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(256, 256)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.discriminator = nn.Sequential(
            sn(nn.Linear(512, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, output_dim)),
        )

    def forward(self, x, z):
        hx = self.x_discrimination(x)
        hz = self.z_discrimination(z)
        out = self.discriminator(torch.cat([hx, hz], dim=1))
        return out


class EncoderFT(nn.Module):
    def __init__(self, latent_dim, channels=1024):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ELU(inplace=True),

            nn.Linear(channels // 2, channels // 4),
            nn.ELU(inplace=True),

            nn.Linear(channels // 4, channels // 8),
            nn.ELU(inplace=True),

            nn.Linear(channels // 8, latent_dim),
        )

    def forward(self, x):
        return self.model(x)


class DecoderFT(nn.Module):
    def __init__(self, latent_dim, channels=1024):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, channels // 8),
            nn.ELU(inplace=True),

            nn.Linear(channels // 8, channels // 4),
            nn.ELU(inplace=True),

            nn.Linear(channels // 4, channels // 2),
            nn.ELU(inplace=True),

            nn.Linear(channels // 2, channels),
        )

    def forward(self, z):
        return self.model(z)


class XZDiscriminatorFT(nn.Module):
    def __init__(self, latent_dim, channels=1024, output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.x_discrimination = nn.Sequential(
            sn(nn.Linear(channels, channels // 2)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels // 2, channels // 4)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels // 4, channels // 8)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(channels // 8, channels // 8)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.z_discrimination = nn.Sequential(
            sn(nn.Linear(latent_dim, 128)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(128, 128)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.discriminator = nn.Sequential(
            sn(nn.Linear(256, 512)),
            nn.LeakyReLU(0.2, inplace=True),

            # sn(nn.Linear(512, 512)),
            # nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(512, output_dim)),
        )

    def forward(self, x, z):
        hx = self.x_discrimination(x)
        hz = self.z_discrimination(z)
        out = self.discriminator(torch.cat([hx, hz], dim=1))
        return out
