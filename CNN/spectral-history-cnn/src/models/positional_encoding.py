from __future__ import annotations

from typing import Iterable

import torch


def build_frequency_positional_encoding(
    height: int,
    width: int,
    lambdas: Iterable[int],
    axis_sigma: float,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    width_rfft = width // 2 + 1
    u = torch.fft.rfftfreq(width, d=1.0, device=device).to(dtype)
    v = torch.fft.fftfreq(height, d=1.0, device=device).to(dtype)
    v = torch.fft.fftshift(v)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

    if v_grid.shape != (height, width_rfft):
        raise RuntimeError(f"Unexpected positional grid shape: {v_grid.shape}")

    radius = torch.sqrt(u_grid**2 + v_grid**2 + 1e-12)
    theta = torch.atan2(v_grid, u_grid)

    channels = [
        u_grid,
        v_grid,
        radius,
        torch.cos(theta),
        torch.sin(theta),
    ]

    axis_u = torch.exp(-0.5 * (u_grid / float(axis_sigma)) ** 2)
    axis_v = torch.exp(-0.5 * (v_grid / float(axis_sigma)) ** 2)
    channels.extend([axis_u, axis_v])

    for lam in lambdas:
        lam = float(lam)
        channels.extend(
            [
                torch.sin(2 * torch.pi * lam * u_grid),
                torch.cos(2 * torch.pi * lam * u_grid),
                torch.sin(2 * torch.pi * lam * v_grid),
                torch.cos(2 * torch.pi * lam * v_grid),
                torch.sin(2 * torch.pi * lam * radius),
                torch.cos(2 * torch.pi * lam * radius),
            ]
        )

    return torch.stack(channels, dim=0)
