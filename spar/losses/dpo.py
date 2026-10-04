"""
DPO Loss (Direct Preference Optimization)

Learns from preference pairs (chosen vs rejected) without explicit reward modeling.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from .base import BasePolicyLoss, LossOutput


class DPOLoss(BasePolicyLoss):
    """
    DPO: Direct Preference Optimization.

    Given (chosen, rejected) pairs, optimizes:
        L = -log sigmoid(beta * (log pi(chosen)/pi_ref(chosen) - log pi(rejected)/pi_ref(rejected)))

    Adapted to BasePolicyLoss interface:
        - log_probs = chosen policy log probs
        - old_log_probs = rejected policy log probs
        - ref_log_probs = chosen reference log probs
        - rejected_ref_log_probs (kwarg) = rejected reference log probs
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.beta = config.get("beta", 0.1)
        self.label_smoothing = config.get("label_smoothing", 0.0)

    def compute_preference_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> tuple:
        """
        Core DPO preference loss computation.

        Args:
            chosen_logps: Sum of log probs for chosen sequences [batch]
            rejected_logps: Sum of log probs for rejected sequences [batch]
            ref_chosen_logps: Reference model log probs for chosen [batch]
            ref_rejected_logps: Reference model log probs for rejected [batch]

        Returns:
            (loss, metrics_dict)
        """
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)

        logits = chosen_rewards - rejected_rewards

        if self.label_smoothing > 0:
            loss = (
                -self.label_smoothing * F.logsigmoid(-logits)
                - (1 - self.label_smoothing) * F.logsigmoid(logits)
            )
        else:
            loss = -F.logsigmoid(logits)

        loss = loss.mean()

        with torch.no_grad():
            accuracy = (logits > 0).float().mean()
            margin = logits.mean()

        metrics = {
            "dpo/chosen_reward_mean": chosen_rewards.mean().item(),
            "dpo/rejected_reward_mean": rejected_rewards.mean().item(),
            "dpo/margin": margin.item(),
            "dpo/accuracy": accuracy.item(),
            "dpo/loss": loss.item(),
        }

        return loss, metrics

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
        Compute DPO loss via BasePolicyLoss interface.

        Mapping:
            log_probs = chosen policy log probs [batch, seq_len]
            old_log_probs = rejected policy log probs [batch, seq_len]
            ref_log_probs = chosen reference log probs [batch, seq_len]
            kwargs['rejected_ref_log_probs'] = rejected ref log probs [batch, seq_len]
            kwargs['rejected_mask'] = mask for rejected sequences [batch, seq_len]
        """
        rejected_ref_log_probs = kwargs.get("rejected_ref_log_probs")
        rejected_mask = kwargs.get("rejected_mask", mask)

        if rejected_ref_log_probs is None:
            rejected_ref_log_probs = old_log_probs.detach()

        # Compute sequence-level log probs (sum over tokens)
        chosen_logps = (log_probs * mask).sum(dim=-1)
        rejected_logps = (old_log_probs * rejected_mask).sum(dim=-1)
        ref_chosen_logps = (ref_log_probs * mask).sum(dim=-1)
        ref_rejected_logps = (rejected_ref_log_probs * rejected_mask).sum(dim=-1)

        loss, metrics = self.compute_preference_loss(
            chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
        )

        # KL for monitoring (not added to loss in DPO)
        kl_loss = self.compute_kl_divergence(log_probs, ref_log_probs, mask)

        metrics["dpo/kl"] = kl_loss.item()

        return LossOutput(
            loss=loss,
            policy_loss=loss,
            kl_loss=kl_loss.detach(),
            metrics=metrics,
        )
