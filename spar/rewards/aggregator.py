"""
Reward Aggregator

Combines multiple reward components with GDPO-style normalization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import torch

from .base import BaseRewardComponent, RewardOutput
from .outcome import OutcomeReward, BeliefShiftOutcomeReward
from .critic import CriticReward, PlanValueReward
from .belief import BeliefShiftReward, PerTurnBeliefReward
from .diversity import PlanDiversityReward, GroupDiversityReward
from .format import FormatReward, TagPresenceReward
from ..config import RewardConfig


@dataclass
class AggregatedRewardOutput:
    """Output from reward aggregation."""

    # Final combined rewards
    rewards: torch.Tensor  # [batch_size]
    advantages: torch.Tensor  # [batch_size] - group-relative

    # Per-component breakdown
    component_rewards: Dict[str, torch.Tensor]
    normalized_rewards: Dict[str, torch.Tensor]

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return self.rewards.mean().item()


class HybridRewardAggregator:
    """
    Hybrid reward aggregator combining GDPO and GRPO normalization.

    Process:
    1. Compute each reward component
    2. Per-component normalization (GDPO-style) - prevents collapse
    3. Weighted combination
    4. Group-relative centering (GRPO-style) - for advantage computation

    This approach:
    - Preserves signal from each reward component
    - Prevents any single component from dominating
    - Produces stable advantages for policy updates
    """

    def __init__(
        self,
        config: RewardConfig,
        critic=None,
        embedding_model=None,
        tokenizer=None,
        device: str = "cuda:0",
    ):
        self.config = config
        self.device = device

        # Create reward components based on config
        self.components: Dict[str, BaseRewardComponent] = {}

        # R_win: Outcome reward
        if config.outcome_weight > 0:
            self.components["outcome"] = OutcomeReward(
                weight=config.outcome_weight,
                for_win_bonus=config.for_win_bonus,
                normalize=config.normalize_per_component,
            )

        # R_crit: Critic-based shaping
        if config.critic_weight > 0 and critic is not None:
            self.components["critic"] = CriticReward(
                critic=critic,
                weight=config.critic_weight,
                gamma=config.critic_gamma,
                normalize=config.normalize_per_component,
            )

        # R_bel: Belief dynamics
        if config.belief_weight > 0:
            self.components["belief"] = BeliefShiftReward(
                weight=config.belief_weight,
                normalize=config.normalize_per_component,
            )

        # R_div: Plan diversity
        if config.diversity_weight > 0:
            self.components["diversity"] = PlanDiversityReward(
                weight=config.diversity_weight,
                embedding_model=embedding_model,
                tokenizer=tokenizer,
                device=device,
                normalize=config.normalize_per_component,
            )

        # R_fmt: Format adherence
        if config.format_weight > 0:
            self.components["format"] = FormatReward(
                weight=config.format_weight,
                normalize=config.normalize_per_component,
            )

        self.eps = config.normalization_eps

    def compute(
        self,
        debate_results: List,
        for_actions: Optional[List[List]] = None,
        states: Optional[List[List[str]]] = None,
        position: str = "FOR",
        **kwargs,
    ) -> AggregatedRewardOutput:
        """
        Compute aggregated rewards for a batch of debates.

        Args:
            debate_results: List of DebateResult objects
            for_actions: Optional list of FOR actions per debate
            states: Optional list of state texts per debate
            position: Position to compute rewards for

        Returns:
            AggregatedRewardOutput with combined rewards and advantages
        """
        component_rewards = {}
        normalized_rewards = {}
        all_metrics = {}

        # Step 1: Compute each component
        for name, component in self.components.items():
            output = component.compute(
                debate_results=debate_results,
                for_actions=for_actions,
                states=states,
                position=position,
                **kwargs,
            )

            component_rewards[name] = output.rewards

            # Step 2: Per-component normalization (GDPO-style)
            if self.config.normalize_per_component:
                normalized_rewards[name] = component.normalize_rewards(output.rewards)
            else:
                normalized_rewards[name] = output.rewards

            # Collect metrics
            for key, value in output.metrics.items():
                all_metrics[key] = value

        # Step 3: Weighted combination
        combined = torch.zeros(len(debate_results))
        for name, norm_rewards in normalized_rewards.items():
            weight = self.components[name].weight
            combined = combined + weight * norm_rewards

        # Step 4: Group-relative centering (GRPO-style)
        advantages = combined - combined.mean()

        # Optionally normalize advantages
        if self.config.normalize_per_component:
            std = advantages.std() + self.eps
            advantages = advantages / std

        # Add aggregation metrics
        all_metrics["aggregator/combined_mean"] = combined.mean().item()
        all_metrics["aggregator/combined_std"] = combined.std().item()
        all_metrics["aggregator/advantage_mean"] = advantages.mean().item()
        all_metrics["aggregator/advantage_std"] = advantages.std().item()

        return AggregatedRewardOutput(
            rewards=combined,
            advantages=advantages,
            component_rewards=component_rewards,
            normalized_rewards=normalized_rewards,
            metrics=all_metrics,
        )

    def compute_for_group(
        self,
        group_results: List[List],
        group_actions: List[List[List]],
        **kwargs,
    ) -> List[AggregatedRewardOutput]:
        """
        Compute rewards for GRPO group (K trajectories per topic).

        Args:
            group_results: [batch, K] debate results
            group_actions: [batch, K, turns] actions

        Returns:
            List of AggregatedRewardOutput per topic
        """
        outputs = []

        for topic_results, topic_actions in zip(group_results, group_actions):
            output = self.compute(
                debate_results=topic_results,
                for_actions=topic_actions,
                **kwargs,
            )
            outputs.append(output)

        return outputs

    def get_weights(self) -> Dict[str, float]:
        """Get current component weights."""
        return {name: comp.weight for name, comp in self.components.items()}

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update component weights."""
        for name, weight in weights.items():
            if name in self.components:
                self.components[name].weight = weight

    def reset_stats(self) -> None:
        """Reset running statistics for all components."""
        for component in self.components.values():
            component.reset_stats()


def create_aggregator(
    config: RewardConfig,
    critic=None,
    embedding_model=None,
    tokenizer=None,
    device: str = "cuda:0",
) -> HybridRewardAggregator:
    """Factory function to create reward aggregator."""
    return HybridRewardAggregator(
        config=config,
        critic=critic,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        device=device,
    )


# Registry of reward components
REWARD_REGISTRY = {
    "outcome": OutcomeReward,
    "belief_shift_outcome": BeliefShiftOutcomeReward,
    "critic": CriticReward,
    "plan_value": PlanValueReward,
    "belief": BeliefShiftReward,
    "per_turn_belief": PerTurnBeliefReward,
    "diversity": PlanDiversityReward,
    "group_diversity": GroupDiversityReward,
    "format": FormatReward,
    "tags": TagPresenceReward,
}


def create_reward_component(
    name: str,
    weight: float = 1.0,
    **kwargs,
) -> BaseRewardComponent:
    """Create a reward component by name."""
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward component: {name}. Available: {list(REWARD_REGISTRY.keys())}")

    return REWARD_REGISTRY[name](weight=weight, **kwargs)
