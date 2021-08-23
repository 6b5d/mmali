import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim, channels=3, num_features=32):
        super().__init__()

        self.model = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(channels, num_features, 4, 2, 1),
            nn.ReLU(inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1),
            nn.ReLU(inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1),
            nn.ReLU(inplace=True),

            # 4x4 -> 1x1
            nn.Conv2d(num_features * 4, latent_dim, 4, 1, 0),
            nn.Flatten(start_dim=1),
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, channels=3, num_features=32):
        super().__init__()

        self.model = nn.Sequential(
            nn.Unflatten(1, (latent_dim, 1, 1)),

            # 1x1 -> 4x4
            nn.ConvTranspose2d(latent_dim, num_features * 4, 4, 1, 0),
            nn.ReLU(inplace=True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1),
            nn.ReLU(inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1),
            nn.ReLU(inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(num_features, channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.model(z)


class XZDiscriminator(nn.Module):
    def __init__(self, latent_dim, channels=3, num_features=32, output_dim=1, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.x_discrimination = nn.Sequential(
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

        self.z_discrimination = nn.Sequential(
            sn(nn.Linear(latent_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(256, 256)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.joint_discriminator = nn.Sequential(
            sn(nn.Linear(512, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, 1024)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Linear(1024, output_dim)),
        )

    def forward(self, x, z):
        hx = self.x_discrimination(x)
        hz = self.z_discrimination(z)
        out = self.joint_discriminator(torch.cat([hx, hz], dim=1))
        return out


class XDiscriminationFeature(nn.Module):
    def __init__(self, channels=3, num_features=32, output_dim=256, spectral_norm=True):
        super().__init__()

        sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
        self.model = nn.Sequential(
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
            sn(nn.Conv2d(num_features * 4, output_dim, 4, 1, 0)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(start_dim=1),
        )

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
            nn.Dropout(0.2),

            sn(nn.Linear(256, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )

        self.joint_discriminator = nn.Sequential(
            sn(nn.Linear(2 * hidden_dim, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            sn(nn.Linear(1024, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            sn(nn.Linear(1024, output_dim)),
        )

    def forward(self, x, z):
        hx = self.x_feature(x)
        hz = self.z_feature(z)
        return self.joint_discriminator(torch.cat([hx, hz], dim=1))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, dilation=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.linear = nn.Linear(in_features=128, out_features=10, bias=True)  # 10 is the number of classes (=digits)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.dropout(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.dropout(h)
        h = self.relu(h)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.dropout(h)
        h = self.relu(h)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.dropout(h)
        h = self.relu(h)

        h = h.view(h.size(0), -1)
        out = self.linear(h)

        return out
