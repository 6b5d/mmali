import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim,
                 img_shape=(1, 28, 28), hidden_dim=400):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(int(np.prod(img_shape)), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim,
                 img_shape=(1, 28, 28), hidden_dim=400):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, int(np.prod(img_shape))),
            nn.Sigmoid(),
            nn.Unflatten(1, img_shape),
        )

    def forward(self, z):
        return self.model(z)


class XZDiscriminator(nn.Module):
    def __init__(self, latent_dim,
                 img_shape=(1, 28, 28), hidden_dim=400, extra_layers=0,
                 output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        layers = [
            sn(nn.Linear(int(np.prod(img_shape)) + latent_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(extra_layers):
            layers += [
                sn(nn.Linear(hidden_dim, hidden_dim)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [
            sn(nn.Linear(hidden_dim, output_dim)),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x, z):
        return self.model(torch.cat([x.view(x.size(0), -1), z], dim=1))


class XZDiscriminator2(nn.Module):
    def __init__(self, latent_dim,
                 img_shape=(1, 28, 28), output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.x_discrimination = nn.Sequential(
            nn.Flatten(start_dim=1),

            sn(nn.Linear(int(np.prod(img_shape)), 256)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.z_discrimination = nn.Sequential(
            sn(nn.Linear(latent_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.joint_discriminator = nn.Sequential(
            sn(nn.Linear(512, 512)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(512, output_dim)),
        )

    def forward(self, x, z):
        hx = self.x_discrimination(x)
        hz = self.z_discrimination(z)
        out = self.joint_discriminator(torch.cat([hx, hz], dim=1))
        return out


class XDiscriminationFeature(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), hidden_dim=400, extra_layers=0,
                 output_dim=256, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        layers = [
            nn.Flatten(start_dim=1),

            sn(nn.Linear(int(np.prod(img_shape)), hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for i in range(extra_layers):
            layers += [
                sn(nn.Linear(hidden_dim, hidden_dim)),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers += [
            sn(nn.Linear(hidden_dim, output_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class XFeatureZDiscriminator(nn.Module):
    def __init__(self, x_feature, latent_dim, hidden_dim=256, output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.x_feature = x_feature

        self.z_feature = nn.Sequential(
            sn(nn.Linear(latent_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.joint_discriminator = nn.Sequential(
            sn(nn.Linear(2 * hidden_dim, 512)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(512, output_dim)),
        )

    def forward(self, x, z):
        hx = self.x_feature(x)
        hz = self.z_feature(z)
        return self.joint_discriminator(torch.cat([hx, hz], dim=1))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return x
