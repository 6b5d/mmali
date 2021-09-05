import torch
import torch.nn as nn


class EncoderFT(nn.Module):
    def __init__(self, latent_dim, channels=2048):
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
    def __init__(self, latent_dim, channels=2048):
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
    def __init__(self, latent_dim, channels=2048, output_dim=1, spectral_norm=True):
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
            sn(nn.Linear(latent_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(256, 256)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.discriminator = nn.Sequential(
            sn(nn.Linear(512, 512)),
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
