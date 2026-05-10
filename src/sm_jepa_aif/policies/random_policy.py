"""Minimal policies for initial exploration experiments."""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    """Uniform random policy over the discrete action set."""

    def __init__(self, num_actions: int = 5, seed: int | None = None) -> None:
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def act(self) -> int:
        return int(self.rng.integers(0, self.num_actions))
