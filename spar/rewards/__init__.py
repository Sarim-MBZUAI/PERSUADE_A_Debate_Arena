"""SPAR Rewards Module."""

from .base import BaseRewardComponent, RewardOutput
from .outcome import OutcomeReward, BeliefShiftOutcomeReward
from .critic import CriticReward, PlanValueReward
from .belief import BeliefShiftReward, PerTurnBeliefReward
from .diversity import PlanDiversityReward, GroupDiversityReward
from .format import FormatReward, TagPresenceReward
from .aggregator import (
    HybridRewardAggregator,
    AggregatedRewardOutput,
    create_aggregator,
    create_reward_component,
    REWARD_REGISTRY,
)

__all__ = [
    # Base
    "BaseRewardComponent",
    "RewardOutput",
    # Outcome
    "OutcomeReward",
    "BeliefShiftOutcomeReward",
    # Critic
    "CriticReward",
    "PlanValueReward",
    # Belief
    "BeliefShiftReward",
    "PerTurnBeliefReward",
    # Diversity
    "PlanDiversityReward",
    "GroupDiversityReward",
    # Format
    "FormatReward",
    "TagPresenceReward",
    # Aggregator
    "HybridRewardAggregator",
    "AggregatedRewardOutput",
    "create_aggregator",
    "create_reward_component",
    "REWARD_REGISTRY",
]
