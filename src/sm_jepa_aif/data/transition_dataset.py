"""Transition dataset API for training sensorimotor prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .rollouts import RolloutEpisode, RolloutLogger


@dataclass(frozen=True)
class TransitionSample:
    """One action-conditioned observation transition."""

    observation: torch.Tensor
    action: torch.Tensor
    next_observation: torch.Tensor
    fixation: torch.Tensor
    next_fixation: torch.Tensor
    image_index: torch.Tensor
    label: torch.Tensor


class TransitionDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of `(o_t, a_t, o_{t+1})` transitions extracted from rollouts."""

    def __init__(self, transitions: dict[str, np.ndarray]) -> None:
        self.transitions = transitions

    @classmethod
    def from_npz(cls, path: str | Path) -> "TransitionDataset":
        archive = np.load(Path(path))
        transitions = {
            key: archive[key]
            for key in archive.files
        }
        return cls(transitions=transitions)

    @classmethod
    def from_rollout_logger(cls, logger: RolloutLogger) -> "TransitionDataset":
        return cls.from_episodes(logger.episodes)

    @classmethod
    def from_episodes(cls, episodes: list[RolloutEpisode]) -> "TransitionDataset":
        observations: list[np.ndarray] = []
        actions: list[int] = []
        next_observations: list[np.ndarray] = []
        fixations: list[tuple[int, int]] = []
        next_fixations: list[tuple[int, int]] = []
        image_indices: list[int] = []
        labels: list[int] = []

        for episode in episodes:
            label = -1 if episode.label is None else int(episode.label)
            for step in episode.steps:
                observations.append(step.observation_before)
                actions.append(step.action)
                next_observations.append(step.observation_after)
                fixations.append(step.fixation_before)
                next_fixations.append(step.fixation_after)
                image_indices.append(episode.image_index)
                labels.append(label)

        glimpse_shape = episodes[0].initial_observation.shape if episodes else (0, 0)
        transitions = {
            "observation": np.stack(observations).astype(np.float32) if observations else np.empty((0, *glimpse_shape), dtype=np.float32),
            "action": np.asarray(actions, dtype=np.int64),
            "next_observation": np.stack(next_observations).astype(np.float32) if next_observations else np.empty((0, *glimpse_shape), dtype=np.float32),
            "fixation": np.asarray(fixations, dtype=np.int64).reshape(-1, 2),
            "next_fixation": np.asarray(next_fixations, dtype=np.int64).reshape(-1, 2),
            "image_index": np.asarray(image_indices, dtype=np.int64),
            "label": np.asarray(labels, dtype=np.int64),
        }
        return cls(transitions=transitions)

    def __len__(self) -> int:
        return int(self.transitions["action"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(value[index]).clone()
            for key, value in self.transitions.items()
        }

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self.transitions.items()}

    def save_npz(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **self.as_arrays())

    def action_histogram(self) -> dict[int, int]:
        if len(self) == 0:
            return {}

        values, counts = np.unique(self.transitions["action"], return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts, strict=True)}

    def summary(self) -> dict[str, Any]:
        return {
            "num_transitions": len(self),
            "glimpse_shape": list(self.transitions["observation"].shape[1:]),
            "action_histogram": self.action_histogram(),
        }
