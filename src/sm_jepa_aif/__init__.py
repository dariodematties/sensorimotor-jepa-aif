"""Top-level package for the sensorimotor JEPA + active inference project."""

from .analysis import evaluate_linear_probe, evaluate_representation, load_checkpoint_models
from .data import RolloutEpisode, RolloutLogger, RolloutStep, TransitionDataset, TransitionSample
from .envs import MNISTGlimpseEnv, StepResult
from .models import ActionConditionedPredictor, GlimpseEncoder

__all__ = [
    "ActionConditionedPredictor",
    "GlimpseEncoder",
    "MNISTGlimpseEnv",
    "RolloutEpisode",
    "RolloutLogger",
    "RolloutStep",
    "StepResult",
    "TransitionDataset",
    "TransitionSample",
    "evaluate_linear_probe",
    "evaluate_representation",
    "load_checkpoint_models",
]
