"""
Format Adherence Reward (R_fmt)

Verifiable reward for following expected output format.
"""

from typing import List
import re
import torch

from .base import BaseRewardComponent, RewardOutput
from ..debate.action import DebateAction
from ..utils.text import validate_plan_format, validate_utterance_format


class FormatReward(BaseRewardComponent):
    """
    R_fmt: Format adherence reward.

    Checks if generated outputs follow the expected format:
    - Plan has <plan>...</plan> tags
    - Plan has 3-6 bullet points
    - Utterance has <debate>...</debate> tags
    - Utterance has 3-5 sentences

    This is a verifiable reward that can be computed without
    external models or API calls.
    """

    def __init__(
        self,
        weight: float = 0.05,
        plan_weight: float = 0.5,
        utterance_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(weight=weight, name="format", **kwargs)
        self.plan_weight = plan_weight
        self.utterance_weight = utterance_weight

    def compute(
        self,
        debate_results: List,
        for_actions: List[List[DebateAction]] = None,
        **kwargs,
    ) -> RewardOutput:
        """
        Compute format adherence rewards.

        Args:
            debate_results: List of completed debates
            for_actions: List of FOR actions per debate

        Returns:
            RewardOutput with format rewards
        """
        rewards = []
        plan_scores = []
        utterance_scores = []

        for i, result in enumerate(debate_results):
            actions = for_actions[i] if for_actions else result.for_actions

            trajectory_reward = 0.0
            for action in actions:
                # Score plan format
                plan_score = self._score_plan(action.plan if hasattr(action, 'plan') else "")
                plan_scores.append(plan_score)

                # Score utterance format
                utterance_score = self._score_utterance(action.utterance)
                utterance_scores.append(utterance_score)

                # Combined score for this turn
                turn_score = (
                    self.plan_weight * plan_score +
                    self.utterance_weight * utterance_score
                )
                trajectory_reward += turn_score

            # Average across turns
            avg_reward = trajectory_reward / len(actions) if actions else 0.0
            rewards.append(avg_reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        return RewardOutput(
            rewards=rewards_tensor,
            metrics={
                f"{self.name}/mean": rewards_tensor.mean().item(),
                f"{self.name}/avg_plan_score": sum(plan_scores) / len(plan_scores) if plan_scores else 0,
                f"{self.name}/avg_utterance_score": sum(utterance_scores) / len(utterance_scores) if utterance_scores else 0,
            },
        )

    def _score_plan(self, plan: str) -> float:
        """Score plan format (0-1)."""
        if not plan:
            return 0.0

        score = 0.0

        # Check for bullet points (3-6 expected)
        bullets = len(re.findall(r"^[\s]*[-*•]\s", plan, re.MULTILINE))
        if 3 <= bullets <= 6:
            score += 0.4
        elif 2 <= bullets <= 7:
            score += 0.2

        # Check length (100-500 chars ideal)
        length = len(plan)
        if 100 <= length <= 500:
            score += 0.3
        elif 50 <= length <= 700:
            score += 0.15

        # Check for counterpoint keywords
        counter_keywords = ['counter', 'anticipate', 'opponent', 'however', 'address', 'rebut']
        if any(kw in plan.lower() for kw in counter_keywords):
            score += 0.15

        # Check for evidence keywords
        evidence_keywords = ['evidence', 'study', 'research', 'data', 'example', 'case']
        if any(kw in plan.lower() for kw in evidence_keywords):
            score += 0.15

        return min(1.0, score)

    def _score_utterance(self, utterance: str) -> float:
        """Score utterance format (0-1)."""
        if not utterance:
            return 0.0

        score = 0.0

        # Check sentence count (3-5 expected)
        sentences = len(re.split(r"[.!?]+", utterance.strip()))
        if 3 <= sentences <= 5:
            score += 0.5
        elif 2 <= sentences <= 6:
            score += 0.25

        # Check length (reasonable range)
        length = len(utterance)
        if 100 <= length <= 600:
            score += 0.3
        elif 50 <= length <= 800:
            score += 0.15

        # Check for coherent structure (has content)
        if len(utterance.split()) >= 20:
            score += 0.2

        return min(1.0, score)


class TagPresenceReward(BaseRewardComponent):
    """
    Simple reward for having correct XML tags.

    Just checks for <plan>...</plan> and <debate>...</debate> tags.
    """

    def __init__(
        self,
        weight: float = 0.05,
        **kwargs,
    ):
        super().__init__(weight=weight, name="tags", **kwargs)

    def compute(
        self,
        debate_results: List,
        for_actions: List[List[DebateAction]] = None,
        **kwargs,
    ) -> RewardOutput:
        """Compute tag presence rewards."""
        rewards = []

        for i, result in enumerate(debate_results):
            actions = for_actions[i] if for_actions else result.for_actions

            trajectory_reward = 0.0
            for action in actions:
                raw_text = action.raw_text if hasattr(action, 'raw_text') and action.raw_text else ""

                score = 0.0

                # Check for plan tags
                if re.search(r"<plan>.*?</plan>", raw_text, re.DOTALL | re.IGNORECASE):
                    score += 0.5

                # Check for debate tags
                if re.search(r"<debate>.*?</debate>", raw_text, re.DOTALL | re.IGNORECASE):
                    score += 0.5

                trajectory_reward += score

            avg_reward = trajectory_reward / len(actions) if actions else 0.0
            rewards.append(avg_reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        return RewardOutput(
            rewards=rewards_tensor,
            metrics={
                f"{self.name}/mean": rewards_tensor.mean().item(),
            },
        )
