"""
SPAR: Strategic Planning and Argumentative Reinforcement
========================================================

A framework for training LLM debate agents using multi-objective
reinforcement learning with GRPO-style optimization.

Main Components:
- config: Configuration dataclasses
- debate: Debate state, actions, agents, and environment
- rewards: Multi-component reward system
- losses: Policy optimization losses (GRPO, SAPO, GDPO, SPARLoss)
- training: Main training loop with curriculum and self-play
- evaluation: Metrics, Elo ratings, and evaluation tools
- models: Critic model for potential-based shaping

Example Usage:
    from spar import SPARConfig, SPARTrainer, create_trainer

    config = SPARConfig.from_yaml("configs/grpo_4turn.yaml")
    trainer = create_trainer(config, model_path="Qwen/Qwen2.5-7B-Instruct")
    trainer.train(topics, num_epochs=3)
"""

__version__ = "0.1.0"
__author__ = "SPAR Team"

from .config import (
    SPARConfig,
    ModelConfig,
    DebateConfig,
    JudgeConfig,
    RewardConfig,
    TrainingConfig,
    load_config,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "SPARConfig",
    "ModelConfig",
    "DebateConfig",
    "JudgeConfig",
    "RewardConfig",
    "TrainingConfig",
    "load_config",
]
