"""Environment package for sensorimotor JEPA + active inference."""

from .mnist_glimpse_env import MNISTGlimpseEnv, StepResult

__all__ = ["MNISTGlimpseEnv", "StepResult"]
