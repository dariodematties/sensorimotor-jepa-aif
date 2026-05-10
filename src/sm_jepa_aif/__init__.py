"""Top-level package for the sensorimotor JEPA + active inference project."""

from .data import RolloutEpisode, RolloutLogger, RolloutStep, TransitionDataset, TransitionSample
from .envs import MNISTGlimpseEnv, StepResult

__all__ = [
    "MNISTGlimpseEnv",
    "RolloutEpisode",
    "RolloutLogger",
    "RolloutStep",
    "StepResult",
    "TransitionDataset",
    "TransitionSample",
]
