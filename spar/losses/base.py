"""
Base Policy Loss Class for SPAR
================================

Abstract base class for all policy optimization losses (GRPO, SAPO, GDPO, SPARLoss).
Adapted from TFPO codebase.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class LossOutput:
    """Container for loss computation outputs."""
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    metrics: Dict[str, float]


class BasePolicyLoss(ABC, nn.Module):
    """
    Abstract base class for policy optimization losses.

    All losses (GRPO, SAPO, GDPO, SPARLoss) inherit from this class.
    Supports multi-component rewards with hybrid normalization.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # KL regularization coefficient
        self.beta = config.get("beta", 0.1)

        # Clipping parameters
        self.epsilon = config.get("epsilon", 0.2)
        self.epsilon_high = config.get("epsilon_high", self.epsilon)

        # Importance sampling level: "token" or "sequence"
        self.importance_sampling_level = config.get("importance_sampling_level", "token")

    @abstractmethod
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
        Compute the policy optimization loss.

        Args:
            log_probs: Log probabilities from current policy [batch, seq_len]
            old_log_probs: Log probabilities from old policy [batch, seq_len]
            ref_log_probs: Log probabilities from reference policy [batch, seq_len]
            advantages: Computed advantages [batch] or [batch, seq_len]
            mask: Attention mask [batch, seq_len]
            **kwargs: Additional arguments

        Returns:
            LossOutput containing loss and metrics
        """
        pass

    def compute_importance_ratio(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute importance sampling ratio.

        For token-level: r_t = exp(log_pi - log_pi_old)
        For sequence-level: r = exp(mean(log_pi - log_pi_old))
        """
        log_ratio = log_probs - old_log_probs

        if self.importance_sampling_level == "token":
            return torch.exp(log_ratio)

        elif self.importance_sampling_level == "sequence":
            # Sequence-level ratio (GSPO style) - geometric mean
            seq_lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1)
            masked_log_ratio = (log_ratio * mask).sum(dim=-1, keepdim=True) / seq_lengths
            return torch.exp(masked_log_ratio).expand_as(log_ratio)

        else:
            raise ValueError(f"Unknown importance_sampling_level: {self.importance_sampling_level}")

    def compute_kl_divergence(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence from reference policy (reverse KL)."""
        kl = log_probs - ref_log_probs
        masked_kl = (kl * mask).sum() / mask.sum().clamp(min=1)
        return masked_kl

    def normalize_advantages(
        self,
        rewards: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Group-normalize advantages (GRPO style)."""
        mean = rewards.mean()
        std = rewards.std()
        return (rewards - mean) / (std + eps)

    def clip_ratio(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clip importance ratio (PPO/GRPO style).

        For positive advantages: clip to [1-eps, 1+eps_high]
        For negative advantages: clip to [1-eps, 1+eps]
        """
        clip_high = torch.where(
            advantages > 0,
            torch.full_like(ratio, 1.0 + self.epsilon_high),
            torch.full_like(ratio, 1.0 + self.epsilon),
        )
        clip_low = torch.full_like(ratio, 1.0 - self.epsilon)

        # Apply clipping with element-wise bounds
        clipped = torch.max(ratio, clip_low)
        clipped = torch.min(clipped, clip_high)
        return clipped

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> LossOutput:
        """Forward pass - calls compute_loss."""
        return self.compute_loss(
            log_probs, old_log_probs, ref_log_probs,
            advantages, mask, **kwargs
        )
