from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from src.models.positional_encoding import build_frequency_positional_encoding


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectralPositionalCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        height: int = 64,
        width: int = 64,
        lambdas: Iterable[int] = (1, 2, 4, 8, 16, 32),
        axis_sigma: float = 0.03,
        dropout: float = 0.2,
    ):
        super().__init__()
        pos_encoding = build_frequency_positional_encoding(
            height=height,
            width=width,
            lambdas=lambdas,
            axis_sigma=axis_sigma,
            dtype=torch.float32,
        )
        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0), persistent=False)
        in_channels = 1 + int(pos_encoding.shape[0])

        self.net = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(128, int(num_classes)),
        )

    @property
    def input_channels(self) -> int:
        return 1 + int(self.pos_encoding.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True, unbiased=False)
        x = (x - mean) / (std + 1e-6)

        pos = self.pos_encoding.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        x_full = torch.cat([x, pos], dim=1)
        return self.net(x_full)
