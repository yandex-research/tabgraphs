"""
Adapted from https://github.com/yandex-research/tabular-dl-tabr/blob/main/lib/deep.py.
"""

import torch
from torch import nn


class PeriodicEmbeddings(nn.Module):
    def __init__(self, features_dim, frequencies_dim, frequencies_scale):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(features_dim, frequencies_dim) * frequencies_scale)

    def forward(self, x):
        x = 2 * torch.pi * self.frequencies[None, ...] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], axis=-1)

        return x


class NLinear(nn.Module):
    def __init__(self, features_dim, input_dim, output_dim, bias=True):
        super().__init__()

        init_max = 1 / torch.tensor(input_dim).sqrt()
        self.weight = nn.Parameter(torch.Tensor(features_dim, input_dim, output_dim).uniform_(-init_max, init_max))
        self.bias = nn.Parameter(torch.Tensor(features_dim, output_dim).uniform_(-init_max, init_max)) if bias else None

    def forward(self, x):
        x = (x[..., None] * self.weight).sum(axis=-2)
        if self.bias is not None:
            x = x + self.bias

        return x


class PLREmbeddings(nn.Module):
    def __init__(self, features_dim, frequencies_dim, frequencies_scale, embedding_dim, lite=False):
        super().__init__()

        if lite:
            linear_layer = nn.Linear(in_features=frequencies_dim * 2, out_features=embedding_dim)
        else:
            linear_layer = NLinear(features_dim=features_dim, input_dim=frequencies_dim * 2, output_dim=embedding_dim)

        self.plr_embeddings = nn.Sequential(
            PeriodicEmbeddings(features_dim=features_dim, frequencies_dim=frequencies_dim,
                               frequencies_scale=frequencies_scale),
            linear_layer,
            nn.ReLU()
        )

    def forward(self, x):
        return self.plr_embeddings(x)
