"""
Belief Dynamics Reward (R_bel)

Reward based on shifting judge's belief toward agent's position.
"""

from typing import List, Optional
import torch

from .base import BaseRewardComponent, RewardOutput
from ..debate.environment import DebateResult


class BeliefShiftReward(BaseRewardComponent):
    """
    R_bel: Reward for shifting judge's belief.

    Measures change in judge's probability estimate that FOR is winning.
    FOR agent gets positive reward when belief moves toward FOR,
    AGAINST agent gets positive reward when belief moves away from FOR.

    Rewards are computed from intermediate belief queries.
    """

    def __init__(
        self,
        weight: float = 0.2,
        scale: float = 2.0,  # Scale belief shift to [-2, 2] range
        **kwargs,
    ):
        super().__init__(weight=weight, name="belief_shift", **kwargs)
        self.scale = scale

    def compute(
        self,
        debate_results: List[DebateResult],
        position: str = "FOR",
        **kwargs,
    ) -> RewardOutput:
        """
        Compute belief shift rewards.

        Args:
            debate_results: List of completed debates with intermediate_beliefs
            position: Position to compute rewards for ("FOR" or "AGAINST")

        Returns:
            RewardOutput with belief shift rewards
        """
        rewards = []
        total_shifts = []

        for result in debate_results:
            beliefs = result.intermediate_beliefs

            if len(beliefs) < 2:
                # Not enough beliefs to compute shift
                # Fall back to initial vs final
                initial = result.judge_initial_belief or (
                    0.6 if result.judge_initial_position == "FOR" else 0.4
                )
                final = 1.0 if result.for_won else 0.0
                shift = final - initial
            else:
                # Compute cumulative belief shift
                shift = beliefs[-1] - beliefs[0]

            total_shifts.append(shift)

            # Scale and adjust for position
            if position == "FOR":
                # FOR wants belief to increase
                reward = self.scale * shift
            else:
                # AGAINST wants belief to decrease
                reward = -self.scale * shift

            rewards.append(reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        return RewardOutput(
            rewards=rewards_tensor,
            metrics={
                f"{self.name}/mean": rewards_tensor.mean().item(),
                f"{self.name}/avg_shift": sum(total_shifts) / len(total_shifts) if total_shifts else 0,
                f"{self.name}/positive_shifts": sum(1 for s in total_shifts if s > 0) / len(total_shifts),
            },
        )


class PerTurnBeliefReward(BaseRewardComponent):
    """
    Per-turn belief reward for finer-grained feedback.

    Computes reward at each turn based on belief change.
    Returns sum of per-turn rewards.
    """

    def __init__(
        self,
        weight: float = 0.2,
        scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(weight=weight, name="per_turn_belief", **kwargs)
        self.scale = scale

    def compute(
        self,
        debate_results: List[DebateResult],
        position: str = "FOR",
        **kwargs,
    ) -> RewardOutput:
        """Compute per-turn belief rewards."""
        rewards = []
        turn_rewards_all = []

        for result in debate_results:
            beliefs = result.intermediate_beliefs
            trajectory_reward = 0.0
            turn_rewards = []

            if len(beliefs) >= 2:
                for t in range(1, len(beliefs)):
                    delta = beliefs[t] - beliefs[t - 1]

                    if position == "FOR":
                        turn_reward = self.scale * delta
                    else:
                        turn_reward = -self.scale * delta

                    trajectory_reward += turn_reward
                    turn_rewards.append(turn_reward)

            rewards.append(trajectory_reward)
            turn_rewards_all.extend(turn_rewards)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        return RewardOutput(
            rewards=rewards_tensor,
            metrics={
                f"{self.name}/mean": rewards_tensor.mean().item(),
                f"{self.name}/avg_turn_reward": (
                    sum(turn_rewards_all) / len(turn_rewards_all)
                    if turn_rewards_all else 0
                ),
            },
        )
