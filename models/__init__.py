import torch.nn as nn

import utils


class GaussianConditional(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *x):
        return utils.reparameterize(self.module(*x))


class DeterministicConditional(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *x):
        return self.module(*x)


class LinearClassifier(nn.Module):
    def __init__(self, module, input_dim, output_dim):
        super().__init__()
        self.module = module
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(self.module(x))


class SliceLayer(nn.Module):
    def __init__(self, module, slicers):
        super().__init__()
        self.module = module
        self.slicers = slicers

    def forward(self, *x):
        out = self.module(*x)
        return out[self.slicers]
