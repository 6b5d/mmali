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


# class XZDiscriminator2(nn.Module):
#     def __init__(self, latent_dim, channels=3, num_features=32, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#
#         self.x_discrimination = nn.Sequential(
#             # 32x32 -> 16x16
#             sn(nn.Conv2d(channels, num_features, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 16x16 -> 8x8
#             sn(nn.Conv2d(num_features, num_features * 2, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 8x8 -> 4x4
#             sn(nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 4x4 -> 1x1
#             sn(nn.Conv2d(num_features * 4, 256, 4, 1, 0)),
#             nn.ReLU(inplace=True),
#
#             nn.Flatten(start_dim=1),
#         )
#
#         self.x_discriminator = sn(nn.Linear(256, 1))
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
#
#             # sn(nn.Linear(256, 256)),
#             # nn.Dropout(dropout),
#             # nn.LeakyReLU(inplace=True),
#         )
#         self.z_discriminator = sn(nn.Linear(256, output_dim))
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
#     def forward(self, x=None, z=None, return_xz=False):
#         if x is None:
#             return self.z_discriminator(self.z_discrimination(z))
#         if z is None:
#             return self.x_discriminator(self.x_discrimination(x))
#         hx = self.x_discrimination(x)
#         hz = self.z_discrimination(z)
#
#         score_joint = self.joint_discriminator(torch.cat([hx, hz], dim=1))
#         if return_xz:
#             score_x = self.x_discriminator(hx)
#             score_z = self.z_discriminator(hz)
#             return score_joint, score_x, score_z
#
#         return score_joint
#
#
# class XZDiscriminatorLeaky(nn.Module):
#     def __init__(self, latent_dim, channels=3, num_features=32, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#
#         self.x_discrimination = nn.Sequential(
#             # 32x32 -> 16x16
#             sn(nn.Conv2d(channels, num_features, 4, 2, 1)),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # 16x16 -> 8x8
#             sn(nn.Conv2d(num_features, num_features * 2, 4, 2, 1)),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # 8x8 -> 4x4
#             sn(nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1)),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # 4x4 -> 1x1
#             sn(nn.Conv2d(num_features * 4, 256, 4, 1, 0)),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Flatten(start_dim=1),
#         )
#
#         self.z_discrimination = nn.Sequential(
#             sn(nn.Linear(latent_dim, 256)),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             sn(nn.Linear(256, 256)),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.joint_discriminator = nn.Sequential(
#             sn(nn.Linear(512, 1024)),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             sn(nn.Linear(1024, 2048)),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             sn(nn.Linear(2048, output_dim)),
#         )
#
#     def forward(self, x, z):
#         hx = self.x_discrimination(x)
#         hz = self.z_discrimination(z)
#         out = self.joint_discriminator(torch.cat([hx, hz], dim=1))
#         return out


# class XZDiscriminator2(nn.Module):
#     def __init__(self, latent_dim, channels=3, num_features=32, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         # dropout = 0.2
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         self.fc = nn.Sequential(
#             sn(nn.Linear(latent_dim, 32 * 32)),
#             nn.ReLU(inplace=True),
#         )
#         self.discriminator = nn.Sequential(
#             # 32x32 -> 16x16
#             sn(nn.Conv2d(channels + 1, num_features, 4, 2, 1)),
#             # nn.Dropout2d(dropout),
#             nn.ReLU(inplace=True),
#
#             # 16x16 -> 8x8
#             sn(nn.Conv2d(num_features, num_features * 2, 4, 2, 1)),
#             # nn.Dropout2d(dropout),
#             nn.ReLU(inplace=True),
#
#             # 8x8 -> 4x4
#             sn(nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1)),
#             # nn.Dropout2d(dropout),
#             nn.ReLU(inplace=True),
#
#             # 4x4 -> 1x1
#             sn(nn.Conv2d(num_features * 4, num_features * 8, 4, 1, 0)),
#             # nn.Dropout2d(dropout),
#             nn.ReLU(inplace=True),
#
#             sn(nn.Conv2d(num_features * 8, output_dim, 1, 1, 0)),
#             nn.Flatten(start_dim=1),
#         )
#
#     def forward(self, x, z):
#         z = self.fc(z).view(-1, 1, 32, 32)
#         return self.discriminator(torch.cat([x, z], dim=1))
#
#
# class XZDiscriminator3(nn.Module):
#     def __init__(self, latent_dim, channels=3, num_features=32, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         dropout = 0.2
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         self.fc = nn.Sequential(
#             sn(nn.Linear(latent_dim, 32 * 32)),
#             # nn.ReLU(inplace=True),
#         )
#         self.x_discrimination = nn.Sequential(
#             # 32x32 -> 16x16
#             sn(nn.Conv2d(channels + 1, num_features, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 16x16 -> 8x8
#             sn(nn.Conv2d(num_features, num_features * 2, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 8x8 -> 4x4
#             sn(nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 4x4 -> 1x1
#             sn(nn.Conv2d(num_features * 4, num_features * 8, 4, 1, 0)),
#             nn.ReLU(inplace=True),
#
#             nn.Flatten(start_dim=1),
#         )
#
#         self.z_discrimination = nn.Sequential(
#             sn(nn.Linear(latent_dim, 256)),
#             nn.LeakyReLU(inplace=True),
#
#             sn(nn.Linear(256, 256)),
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
#         hx = self.x_discrimination(torch.cat([x, self.fc(z).view(-1, 1, 32, 32)], dim=1))
#         hz = self.z_discrimination(z)
#         out = self.discriminator(torch.cat([hx, hz], dim=1))
#         return out
#
#
# class XZDiscriminator4(nn.Module):
#     def __init__(self, latent_dim, channels=3, num_features=32, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         dropout = 0.2
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#
#         self.x_discrimination = nn.Sequential(
#             # 32x32 -> 16x16
#             sn(nn.Conv2d(channels, num_features, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 16x16 -> 8x8
#             sn(nn.Conv2d(num_features, num_features * 2, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 8x8 -> 4x4
#             sn(nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#
#             # 4x4 -> 1x1
#             sn(nn.Conv2d(num_features * 4, 256, 4, 1, 0)),
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
#         # self.discriminator = nn.Sequential(
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
#         self.discriminator = sn(nn.Bilinear(256, 256, output_dim))
#
#     def forward(self, x, z):
#         hx = self.x_discrimination(x)
#         hz = self.z_discrimination(z)
#         out = self.discriminator(hx, hz)
#         return out


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


#
# class OldClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(500, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 500)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x


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
# class XZDiscriminatorFM(nn.Module):
#     def __init__(self, latent_dim, channels=3, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         slope = 0.01
#         dropout = 0.2
#         self.x_layer1 = nn.Sequential(
#             # 32x32 -> 16x16
#             sn(nn.Conv2d(channels, 32, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#         )
#         self.x_layer2 = nn.Sequential(
#             # 16x16 -> 8x8
#             sn(nn.Conv2d(32, 64, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#         )
#         self.x_layer3 = nn.Sequential(
#             # 8x8 -> 4x4
#             sn(nn.Conv2d(64, 128, 4, 2, 1)),
#             nn.ReLU(inplace=True),
#         )
#         self.x_layer4 = nn.Sequential(
#             # 4x4 -> 1x1
#             sn(nn.Conv2d(128, 256, 4, 1, 0)),
#             nn.ReLU(inplace=True),
#             nn.Flatten(start_dim=1),
#         )
#
#         self.z_layer1 = nn.Sequential(
#             sn(nn.Linear(latent_dim, 256)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(slope, inplace=True),
#         )
#         self.z_layer2 = nn.Sequential(
#             sn(nn.Linear(256, 256)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(slope, inplace=True),
#         )
#
#         self.joint_layer1 = nn.Sequential(
#             sn(nn.Linear(512, 1024)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(slope, inplace=True),
#         )
#         self.joint_layer2 = nn.Sequential(
#             sn(nn.Linear(1024, 1024)),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(slope, inplace=True),
#         )
#         self.joint_layer3 = sn(nn.Linear(1024, output_dim))
#
#     def forward(self, x, z, feature=False):
#         hx = x
#         x_feats = []
#         for i in [1, 2, 3, 4]:
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
#             return h, x_feats[3:] + z_feats + joint_feats[2:]
#
#         return h


#
#
# class XDiscriminator(nn.Module):
#     def __init__(self, channels=3, output_dim=256, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
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
#             sn(nn.Conv2d(128, output_dim, 4, 1, 0)),
#             nn.ReLU(inplace=True),
#
#             nn.Flatten(start_dim=1),
#         ]
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# class ZDiscriminator(nn.Module):
#     def __init__(self, latent_dim, output_dim=256, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         layers = [
#             sn(nn.Linear(latent_dim, 256)),
#             nn.ReLU(inplace=True),
#
#             sn(nn.Linear(256, output_dim)),
#             nn.ReLU(inplace=True),
#         ]
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, z):
#         return self.model(z)
#
#
# class XZDiscriminatorShared(nn.Module):
#     def __init__(self, x_discriminator, z_discriminator,
#                  input_dim=512, output_dim=1, spectral_norm=True):
#         super().__init__()
#         self.x_discriminator = x_discriminator
#         self.z_discriminator = z_discriminator
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         layers = [
#             sn(nn.Linear(input_dim, 512)),
#             nn.ReLU(inplace=True),
#
#             sn(nn.Linear(512, 512)),
#             nn.ReLU(inplace=True),
#
#             sn(nn.Linear(512, output_dim)),
#         ]
#
#         self.joint_discriminator = nn.Sequential(*layers)
#
#     def forward(self, x, z):
#         hx = self.x_discriminator(x)
#         hz = self.z_discriminator(z)
#         return self.joint_discriminator(torch.cat([hx, hz], dim=1))
#
#
# class BilinearXZDiscriminator(nn.Module):
#     def __init__(self, latent_dim, channels=3, output_dim=1, spectral_norm=True):
#         super().__init__()
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         layers = [
#             sn(nn.Conv2d(channels, 32, 4, 2, 1)),
#             nn.LeakyReLU(0.01, inplace=True),
#
#             sn(nn.Conv2d(32, 64, 4, 2, 1)),
#             nn.LeakyReLU(0.01, inplace=True),
#
#             sn(nn.Conv2d(64, 128, 4, 2, 1)),
#             nn.LeakyReLU(0.01, inplace=True),
#
#             sn(nn.Conv2d(128, 256, 4, 1, 0)),
#             nn.LeakyReLU(0.01, inplace=True),
#
#             nn.Flatten(start_dim=1),
#         ]
#         self.x_discriminator = nn.Sequential(*layers)
#
#         layers = [
#             sn(nn.Linear(latent_dim, 256)),
#             nn.LeakyReLU(0.01, inplace=True),
#
#             sn(nn.Linear(256, 256)),
#             nn.LeakyReLU(0.01, inplace=True),
#
#             sn(nn.Linear(256, 256)),
#             nn.LeakyReLU(0.01, inplace=True),
#         ]
#         self.z_discriminator = nn.Sequential(*layers)
#
#         self.joint_discriminator = sn(nn.Bilinear(256, 256, 1))
#
#     def forward(self, x, z):
#         hx = self.x_discriminator(x)
#         hz = self.z_discriminator(z)
#         return self.joint_discriminator(hx, hz)
#
#
# class BilinearXZDiscriminatorShared(nn.Module):
#     def __init__(self, x_discriminator, z_discriminator,
#                  x_dim=256, z_dim=256, output_dim=1, spectral_norm=True):
#         super().__init__()
#         self.x_discriminator = x_discriminator
#         self.z_discriminator = z_discriminator
#
#         sn = nn.utils.spectral_norm if spectral_norm else lambda x: x
#         self.joint_discriminator = sn(nn.Bilinear(x_dim, z_dim, output_dim))
#
#     def forward(self, x, z):
#         hx = self.x_discriminator(x)
#         hz = self.z_discriminator(z)
#         return self.joint_discriminator(hx, hz)
