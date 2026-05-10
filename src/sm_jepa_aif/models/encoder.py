"""Observation encoder for the first sensorimotor JEPA baseline."""

from __future__ import annotations

import torch
from torch import nn


class GlimpseEncoder(nn.Module):
    """Map a single glimpse observation to a latent state vector."""

    def __init__(
        self,
        glimpse_size: int = 7,
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = glimpse_size * glimpse_size

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 2:
            observation = observation.unsqueeze(0)
        return self.network(observation)
