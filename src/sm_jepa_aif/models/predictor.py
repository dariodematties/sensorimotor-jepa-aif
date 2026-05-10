"""Action-conditioned latent predictor for the first JEPA baseline."""

from __future__ import annotations

import torch
from torch import nn


class ActionConditionedPredictor(nn.Module):
    """Predict the next latent state from current latent state and action."""

    def __init__(
        self,
        latent_dim: int = 32,
        num_actions: int = 5,
        action_embed_dim: int = 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_features = self.action_embedding(action.long())
        features = torch.cat([latent, action_features], dim=-1)
        return self.network(features)
