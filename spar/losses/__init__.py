"""SPAR Losses Module.

Policy optimization losses used in the paper:
- GRPO: Group Relative Policy Optimization (token-level, hard clipping)
- DPO: Direct Preference Optimization
"""

from .base import BasePolicyLoss, LossOutput
from .grpo import GRPOLoss, GRPOLossWithEntropy
from .dpo import DPOLoss

__all__ = [
    # Base
    "BasePolicyLoss",
    "LossOutput",
    # GRPO
    "GRPOLoss",
    "GRPOLossWithEntropy",
    # DPO
    "DPOLoss",
]


# Loss registry for factory creation
LOSS_REGISTRY = {
    "grpo": GRPOLoss,
    "grpo_entropy": GRPOLossWithEntropy,
    "dpo": DPOLoss,
}


def create_loss(loss_type: str, config: dict) -> BasePolicyLoss:
    """Factory function to create a loss by type.

    Args:
        loss_type: One of "grpo", "grpo_entropy", "dpo"
        config: Loss configuration dict

    Returns:
        Instantiated loss module
    """
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available: {list(LOSS_REGISTRY.keys())}"
        )

    return LOSS_REGISTRY[loss_type](config)
