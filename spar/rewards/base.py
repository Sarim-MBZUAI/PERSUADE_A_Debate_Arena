"""
Base Reward Component

Abstract base class for all reward components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch


@dataclass
class RewardOutput:
    """Output from a reward component."""

    rewards: torch.Tensor  # [batch_size]
    metrics: Dict[str, float] = field(default_factory=dict)
    details: Optional[Dict[str, Any]] = None

    @property
    def mean(self) -> float:
        return self.rewards.mean().item()

    @property
    def std(self) -> float:
        return self.rewards.std().item()


class BaseRewardComponent(ABC):
    """
    Abstract base class for reward components.

    Each component computes a specific aspect of the reward:
    - R_win: Outcome reward
    - R_crit: Critic-based shaping
    - R_bel: Belief dynamics
    - R_div: Plan diversity
    - R_fmt: Format adherence
    """

    def __init__(
        self,
        weight: float = 1.0,
        name: Optional[str] = None,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        self.weight = weight
        self.name = name or self.__class__.__name__
        self.normalize = normalize
        self.eps = eps

        # Running statistics for normalization
        self._running_mean = 0.0
        self._running_var = 1.0
        self._count = 0
        self._decay = 0.99

    @abstractmethod
    def compute(
        self,
        debate_results: List["DebateResult"],
        **kwargs,
    ) -> RewardOutput:
        """
        Compute rewards for a batch of debates.

        Args:
            debate_results: List of completed debate results
            **kwargs: Additional arguments (for_actions, states, etc.)

        Returns:
            RewardOutput with rewards and metrics
        """
        pass

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Per-component normalization (GDPO-style).

        Normalizes using running statistics to prevent reward collapse.
        """
        if not self.normalize:
            return rewards

        # Update running statistics
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item()

        self._count += 1
        if self._count == 1:
            self._running_mean = batch_mean
            self._running_var = batch_var
        else:
            self._running_mean = (
                self._decay * self._running_mean +
                (1 - self._decay) * batch_mean
            )
            self._running_var = (
                self._decay * self._running_var +
                (1 - self._decay) * batch_var
            )

        # Normalize
        std = (self._running_var + self.eps) ** 0.5
        normalized = (rewards - self._running_mean) / std

        return normalized

    def reset_stats(self) -> None:
        """Reset running statistics."""
        self._running_mean = 0.0
        self._running_var = 1.0
        self._count = 0

    def get_stats(self) -> Dict[str, float]:
        """Get current running statistics."""
        return {
            f"{self.name}/running_mean": self._running_mean,
            f"{self.name}/running_var": self._running_var,
            f"{self.name}/count": self._count,
        }
