"""
GRPO Loss (Group Relative Policy Optimization)

Token-level PPO-style loss with hard clipping.
Adapted from TFPO codebase.
"""

import torch
from typing import Dict, Optional

from .base import BasePolicyLoss, LossOutput


class GRPOLoss(BasePolicyLoss):
    """
    GRPO: Group Relative Policy Optimization.

    Key features:
    - Token-level importance sampling
    - Hard clipping (PPO-style)
    - Group-relative advantage normalization
    - KL penalty to reference policy

    Loss = -E[min(r*A, clip(r)*A)] + beta * KL(pi || pi_ref)
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # GRPO-specific parameters
        self.use_advantage_normalization = config.get("use_advantage_normalization", True)
        self.clip_value_loss = config.get("clip_value_loss", False)

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> LossOutput:
        """
        Compute GRPO loss.

        Args:
            log_probs: Current policy log probs [batch, seq_len]
            old_log_probs: Old policy log probs [batch, seq_len]
            ref_log_probs: Reference policy log probs [batch, seq_len]
            advantages: Computed advantages [batch] or [batch, seq_len]
            mask: Token mask [batch, seq_len]

        Returns:
            LossOutput with loss and metrics
        """
        # Compute importance ratio
        ratio = self.compute_importance_ratio(log_probs, old_log_probs, mask)

        # Expand advantages if sequence-level
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1).expand_as(ratio)

        # Normalize advantages (GRPO-style)
        if self.use_advantage_normalization:
            adv_mean = (advantages * mask).sum() / mask.sum().clamp(min=1)
            adv_std = torch.sqrt(
                ((advantages - adv_mean) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            ).clamp(min=1e-8)
            advantages = (advantages - adv_mean) / adv_std

        # Compute clipped ratio
        clipped_ratio = self.clip_ratio(ratio, advantages)

        # PPO-style objective: min of unclipped and clipped
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages

        # Take minimum (pessimistic bound)
        policy_loss = -torch.min(surr1, surr2)

        # Apply mask and compute mean
        masked_policy_loss = (policy_loss * mask).sum() / mask.sum().clamp(min=1)

        # KL penalty
        kl_loss = self.compute_kl_divergence(log_probs, ref_log_probs, mask)

        # Total loss
        total_loss = masked_policy_loss + self.beta * kl_loss

        # Compute metrics
        with torch.no_grad():
            clip_fraction = ((ratio - clipped_ratio).abs() > 1e-6).float()
            clip_fraction = (clip_fraction * mask).sum() / mask.sum().clamp(min=1)

            approx_kl = 0.5 * ((log_probs - old_log_probs) ** 2 * mask).sum() / mask.sum().clamp(min=1)

        metrics = {
            "grpo/policy_loss": masked_policy_loss.item(),
            "grpo/kl_loss": kl_loss.item(),
            "grpo/total_loss": total_loss.item(),
            "grpo/clip_fraction": clip_fraction.item(),
            "grpo/approx_kl": approx_kl.item(),
            "grpo/ratio_mean": (ratio * mask).sum().item() / mask.sum().clamp(min=1).item(),
            "grpo/advantage_mean": (advantages * mask).sum().item() / mask.sum().clamp(min=1).item(),
        }

        return LossOutput(
            loss=total_loss,
            policy_loss=masked_policy_loss,
            kl_loss=kl_loss,
            metrics=metrics,
        )


class GRPOLossWithEntropy(GRPOLoss):
    """
    GRPO with entropy bonus for exploration.

    Loss = GRPO_loss - entropy_coef * H(pi)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.entropy_coef = config.get("entropy_coef", 0.01)

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> LossOutput:
        """Compute GRPO loss with entropy bonus."""
        # Get base GRPO loss
        output = super().compute_loss(
            log_probs, old_log_probs, ref_log_probs,
            advantages, mask, **kwargs
        )

        # Compute entropy bonus
        # Approximate entropy from log probs
        entropy = -log_probs
        masked_entropy = (entropy * mask).sum() / mask.sum().clamp(min=1)

        # Add entropy bonus (negative because we want to maximize entropy)
        total_loss = output.loss - self.entropy_coef * masked_entropy

        # Update metrics
        output.metrics["grpo/entropy"] = masked_entropy.item()
        output.metrics["grpo/total_loss"] = total_loss.item()

        return LossOutput(
            loss=total_loss,
            policy_loss=output.policy_loss,
            kl_loss=output.kl_loss,
            metrics=output.metrics,
        )
