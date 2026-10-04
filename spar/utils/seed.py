"""
Reproducibility Utilities

Seeding for deterministic experiments.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Note: These may slow down training
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_generator(seed: int) -> torch.Generator:
    """Get a seeded generator for sampling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class SeedManager:
    """
    Manage seeds for different components of training.

    Useful for reproducible GRPO where we want different seeds
    for different plan candidates but consistent across runs.
    """

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.step_counter = 0

    def get_plan_seed(self, topic_idx: int, plan_idx: int) -> int:
        """Get deterministic seed for plan generation."""
        return self.base_seed + topic_idx * 1000 + plan_idx

    def get_step_seed(self) -> int:
        """Get seed for current training step."""
        seed = self.base_seed + self.step_counter
        self.step_counter += 1
        return seed

    def reset(self) -> None:
        """Reset step counter."""
        self.step_counter = 0
