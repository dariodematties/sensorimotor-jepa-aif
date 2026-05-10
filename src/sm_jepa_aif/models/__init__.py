"""Model components for sensorimotor JEPA experiments."""

from .encoder import GlimpseEncoder
from .predictor import ActionConditionedPredictor

__all__ = ["ActionConditionedPredictor", "GlimpseEncoder"]
