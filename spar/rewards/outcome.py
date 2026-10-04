"""
Outcome Reward (R_win)

Binary reward based on final judge verdict.
"""

from typing import List
import torch

from .base import BaseRewardComponent, RewardOutput
from ..debate.environment import DebateResult


class OutcomeReward(BaseRewardComponent):
    """
    R_win: Outcome reward based on final judge verdict.

    Returns:
    - +for_win_bonus (default 1.5) if FOR wins
    - -1.0 if AGAINST wins

    The asymmetric bonus helps overcome the AGAINST advantage
    observed in experiments.
    """

    def __init__(
        self,
        weight: float = 1.0,
        for_win_bonus: float = 1.5,
        **kwargs,
    ):
        super().__init__(weight=weight, name="outcome", **kwargs)
        self.for_win_bonus = for_win_bonus

    def compute(
        self,
        debate_results: List[DebateResult],
        position: str = "FOR",
        **kwargs,
    ) -> RewardOutput:
        """
        Compute outcome rewards for batch of debates.

        Args:
            debate_results: List of completed debates
            position: Which position to compute rewards for ("FOR" or "AGAINST")

        Returns:
            RewardOutput with rewards
        """
        rewards = []

        for result in debate_results:
            if position == "FOR":
                if result.for_won:
                    reward = self.for_win_bonus
                else:
                    reward = -1.0
            else:  # AGAINST
                if result.against_won:
                    reward = 1.0
                else:
                    reward = -1.0

            rewards.append(reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        # Compute metrics
        win_rate = (rewards_tensor > 0).float().mean().item()

        return RewardOutput(
            rewards=rewards_tensor,
            metrics={
                f"{self.name}/win_rate": win_rate,
                f"{self.name}/mean": rewards_tensor.mean().item(),
            },
        )


class BeliefShiftOutcomeReward(BaseRewardComponent):
    """
    Enhanced outcome reward that considers belief shift.

    Gives bonus for:
    - Winning when judge initially disagreed (opinion change)
    - Larger belief shift toward winning position
    """

    def __init__(
        self,
        weight: float = 1.0,
        for_win_bonus: float = 1.5,
        opinion_change_bonus: float = 0.5,
        **kwargs,
    ):
        super().__init__(weight=weight, name="belief_shift_outcome", **kwargs)
        self.for_win_bonus = for_win_bonus
        self.opinion_change_bonus = opinion_change_bonus

    def compute(
        self,
        debate_results: List[DebateResult],
        position: str = "FOR",
        **kwargs,
    ) -> RewardOutput:
        """Compute outcome rewards with belief shift bonus."""
        rewards = []
        opinion_changes = 0

        for result in debate_results:
            reward = 0.0

            if position == "FOR":
                if result.for_won:
                    reward = self.for_win_bonus

                    # Bonus for changing judge's opinion
                    if result.judge_initial_position == "AGAINST":
                        reward += self.opinion_change_bonus
                        opinion_changes += 1
                else:
                    reward = -1.0

                    # Extra penalty for losing after having initial support
                    if result.judge_initial_position == "FOR":
                        reward -= self.opinion_change_bonus * 0.5

            else:  # AGAINST
                if result.against_won:
                    reward = 1.0
                    if result.judge_initial_position == "FOR":
                        reward += self.opinion_change_bonus
                        opinion_changes += 1
                else:
                    reward = -1.0

            rewards.append(reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        return RewardOutput(
            rewards=rewards_tensor,
            metrics={
                f"{self.name}/win_rate": (rewards_tensor > 0).float().mean().item(),
                f"{self.name}/opinion_change_rate": opinion_changes / len(debate_results),
                f"{self.name}/mean": rewards_tensor.mean().item(),
            },
        )
