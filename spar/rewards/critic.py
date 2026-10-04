"""
Critic-Based Reward (R_crit)

Potential-based reward shaping using V(state, plan).
"""

from typing import List, Optional
import torch

from .base import BaseRewardComponent, RewardOutput
from ..debate.environment import DebateResult
from ..debate.action import DebateAction


class CriticReward(BaseRewardComponent):
    """
    R_crit: Critic-based potential shaping reward.

    Uses a trained critic V(state, plan) to provide dense rewards
    via temporal difference:

        r_crit(t) = gamma * V(s_{t+1}, plan_{t+1}) - V(s_t, plan_t)

    This is potential-based shaping which preserves optimal policies.
    """

    def __init__(
        self,
        critic,  # CriticModel
        weight: float = 0.3,
        gamma: float = 0.99,
        **kwargs,
    ):
        super().__init__(weight=weight, name="critic", **kwargs)
        self.critic = critic
        self.gamma = gamma

    def compute(
        self,
        debate_results: List[DebateResult],
        for_actions: Optional[List[List[DebateAction]]] = None,
        states: Optional[List[List[str]]] = None,
        **kwargs,
    ) -> RewardOutput:
        """
        Compute critic-based shaping rewards.

        For each debate, computes temporal difference at each turn
        and sums to get trajectory-level reward.

        Args:
            debate_results: List of completed debates
            for_actions: List of FOR actions per debate [batch, turns]
            states: List of state texts per debate [batch, turns]

        Returns:
            RewardOutput with shaped rewards
        """
        rewards = []
        critic_values = []

        for i, result in enumerate(debate_results):
            trajectory_reward = 0.0
            prev_value = 0.5  # Initial value (neutral)

            # Get actions and states for this debate
            actions = for_actions[i] if for_actions else result.for_actions
            debate_states = states[i] if states else self._extract_states(result)

            for t, (action, state_text) in enumerate(zip(actions, debate_states)):
                # Get critic value for current state-plan pair
                if hasattr(action, 'plan') and action.plan:
                    current_value = self.critic.score(state_text, action.plan)
                else:
                    current_value = self.critic.score(state_text, action.utterance[:100])

                critic_values.append(current_value)

                # TD reward: gamma * V(s') - V(s)
                if t > 0:
                    td_reward = self.gamma * current_value - prev_value
                    trajectory_reward += td_reward

                prev_value = current_value

            # Final TD to terminal value (1 if won, 0 if lost)
            terminal_value = 1.0 if result.for_won else 0.0
            trajectory_reward += self.gamma * terminal_value - prev_value

            rewards.append(trajectory_reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        return RewardOutput(
            rewards=rewards_tensor,
            metrics={
                f"{self.name}/mean": rewards_tensor.mean().item(),
                f"{self.name}/std": rewards_tensor.std().item(),
                f"{self.name}/critic_mean": sum(critic_values) / len(critic_values) if critic_values else 0,
            },
        )

    def _extract_states(self, result: DebateResult) -> List[str]:
        """Extract state texts from debate result."""
        states = []
        history = []

        for entry in result.history:
            if entry.get("agent") == "FOR":
                # Build state text before this entry
                state_text = f"TOPIC: {result.topic}\nPOSITION: FOR\n"
                if history:
                    state_text += "DEBATE SO FAR:\n"
                    for h in history[-6:]:  # Last 3 turns
                        state_text += f"Turn {h.get('turn')} - {h.get('agent')}: {h.get('argument')}\n"
                states.append(state_text)

            history.append(entry)

        return states

    def compute_single(
        self,
        state_text: str,
        plan: str,
    ) -> float:
        """Compute critic value for single state-plan pair."""
        return self.critic.score(state_text, plan)


class PlanValueReward(BaseRewardComponent):
    """
    Simpler critic reward that just returns V(s, plan) value.

    No temporal difference, just raw critic scores.
    Useful for plan selection/ranking.
    """

    def __init__(
        self,
        critic,  # CriticModel
        weight: float = 0.3,
        **kwargs,
    ):
        super().__init__(weight=weight, name="plan_value", **kwargs)
        self.critic = critic

    def compute(
        self,
        debate_results: List[DebateResult],
        for_actions: Optional[List[List[DebateAction]]] = None,
        states: Optional[List[List[str]]] = None,
        **kwargs,
    ) -> RewardOutput:
        """Compute average critic value across trajectory."""
        rewards = []

        for i, result in enumerate(debate_results):
            actions = for_actions[i] if for_actions else result.for_actions
            debate_states = states[i] if states else self._extract_states(result)

            total_value = 0.0
            count = 0

            for action, state_text in zip(actions, debate_states):
                if hasattr(action, 'plan') and action.plan:
                    value = self.critic.score(state_text, action.plan)
                    total_value += value
                    count += 1

            avg_value = total_value / count if count > 0 else 0.5
            rewards.append(avg_value)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        return RewardOutput(
            rewards=rewards_tensor,
            metrics={
                f"{self.name}/mean": rewards_tensor.mean().item(),
            },
        )

    def _extract_states(self, result: DebateResult) -> List[str]:
        """Extract state texts from debate result."""
        states = []
        history = []

        for entry in result.history:
            if entry.get("agent") == "FOR":
                state_text = f"TOPIC: {result.topic}\nPOSITION: FOR\n"
                if history:
                    state_text += "DEBATE SO FAR:\n"
                    for h in history[-6:]:
                        state_text += f"Turn {h.get('turn')} - {h.get('agent')}: {h.get('argument')}\n"
                states.append(state_text)

            history.append(entry)

        return states
