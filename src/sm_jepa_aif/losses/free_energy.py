"""Loss helpers for the first sensorimotor JEPA training baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_prediction_loss(
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
) -> torch.Tensor:
    """Deterministic proxy for latent prediction error.

    The long-term project aims for a probabilistic KL objective. The initial
    baseline uses mean-squared error between predicted and inferred next
    latents so the full training loop can be validated end to end first.
    """

    return F.mse_loss(predicted_latent, target_latent)
