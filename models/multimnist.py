import numpy as np
import torch
import torch.nn as nn


class MultiXDiscriminator(nn.Module):
    def __init__(self, n_modalities, img_shape=(1, 28, 28), output_dim=1, spectral_norm=True):
        super().__init__()
        self.num_modalities = n_modalities

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.x_discriminations = nn.ModuleList()
        for i in range(self.num_modalities):
            self.x_discriminations.add_module(
                'x{}_discrimination'.format(i),
                nn.Sequential(
                    nn.Flatten(start_dim=1),

                    sn(nn.Linear(int(np.prod(img_shape)), 400)),
                    nn.LeakyReLU(0.2, inplace=True),

                    sn(nn.Linear(400, 256)),
                    nn.LeakyReLU(0.2, inplace=True),
                ))

        self.joint_discriminator = nn.Sequential(
            sn(nn.Linear(256 * self.num_modalities, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, output_dim)),
        )

    def forward(self, *inputs):
        hx = []
        for i, x in enumerate(inputs):
            discrimination = getattr(self.x_discriminations, 'x{}_discrimination'.format(i))
            hx.append(discrimination(x))

        out = self.joint_discriminator(torch.cat(hx, dim=1))
        return out


class MultiXZDiscriminator(nn.Module):
    def __init__(self, n_modalities, latent_dim, img_shape=(1, 28, 28), output_dim=1, spectral_norm=True):
        super().__init__()
        self.num_modalities = n_modalities

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.x_discriminations = nn.ModuleList()
        for i in range(self.num_modalities):
            self.x_discriminations.add_module(
                'x{}_discrimination'.format(i),
                nn.Sequential(
                    nn.Flatten(start_dim=1),

                    sn(nn.Linear(int(np.prod(img_shape)), 400)),
                    nn.LeakyReLU(0.2, inplace=True),

                    sn(nn.Linear(400, 256)),
                    nn.LeakyReLU(0.2, inplace=True),
                ))

        self.z_discrimination = nn.Sequential(
            sn(nn.Linear(latent_dim, 400)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(400, 256)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.joint_discriminator = nn.Sequential(
            sn(nn.Linear(256 * (self.num_modalities + 1), 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, output_dim)),
        )

    def forward(self, *inputs):
        hx = []
        for i, x in enumerate(inputs):
            discrimination = getattr(self.x_discriminations, 'x{}_discrimination'.format(i))
            hx.append(discrimination(x))

        out = self.joint_discriminator(torch.cat(hx, dim=1))
        return out
