"""SPAR Configuration Module."""

from .base import (
    SPARConfig,
    ModelConfig,
    DebateConfig,
    JudgeConfig,
    RewardConfig,
    TrainingConfig,
    load_config,
)

__all__ = [
    "SPARConfig",
    "ModelConfig",
    "DebateConfig",
    "JudgeConfig",
    "RewardConfig",
    "TrainingConfig",
    "load_config",
]
