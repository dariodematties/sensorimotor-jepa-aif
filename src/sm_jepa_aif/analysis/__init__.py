"""Representation diagnostics for sensorimotor JEPA baselines."""

from .latent_diagnostics import (
    build_diagnostic_notes,
    evaluate_linear_probe,
    evaluate_representation,
    load_checkpoint_models,
)

__all__ = [
    "build_diagnostic_notes",
    "evaluate_linear_probe",
    "evaluate_representation",
    "load_checkpoint_models",
]
