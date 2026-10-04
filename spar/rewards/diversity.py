"""
Plan Diversity Reward (R_div)

Encourages exploration of diverse strategic plans.
"""

from typing import List, Optional
import torch
import torch.nn.functional as F

from .base import BaseRewardComponent, RewardOutput
from ..debate.action import DebateAction


class PlanDiversityReward(BaseRewardComponent):
    """
    R_div: Reward for diverse plans.

    Measures semantic diversity of plans using embedding similarity.
    Rewards plans that are different from previous plans in the group
    or in the trajectory.

    Uses cosine distance: 1 - cosine_similarity.
    """

    def __init__(
        self,
        weight: float = 0.1,
        embedding_model=None,  # Model to get plan embeddings
        tokenizer=None,
        device: str = "cuda:0",
        **kwargs,
    ):
        super().__init__(weight=weight, name="diversity", **kwargs)
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.device = device

        # Cache for plan embeddings
        self._embedding_cache = {}

    def compute(
        self,
        debate_results: List,
        for_actions: Optional[List[List[DebateAction]]] = None,
        **kwargs,
    ) -> RewardOutput:
        """
        Compute diversity rewards.

        For each debate, computes average distance of each plan
        to other plans in the trajectory.

        Args:
            debate_results: List of completed debates
            for_actions: List of FOR actions per debate

        Returns:
            RewardOutput with diversity rewards
        """
        rewards = []
        all_diversities = []

        for i, result in enumerate(debate_results):
            actions = for_actions[i] if for_actions else result.for_actions

            # Collect all plans in this trajectory
            plans = [a.plan for a in actions if hasattr(a, 'plan') and a.plan]

            if len(plans) < 2:
                # Not enough plans for diversity
                rewards.append(0.0)
                continue

            # Get embeddings
            embeddings = self._get_embeddings(plans)

            # Compute pairwise distances
            diversities = []
            for j, emb in enumerate(embeddings):
                other_embs = embeddings[:j] + embeddings[j+1:]
                if other_embs:
                    other_tensor = torch.stack(other_embs)
                    similarities = F.cosine_similarity(emb.unsqueeze(0), other_tensor)
                    avg_distance = 1 - similarities.mean().item()
                    diversities.append(avg_distance)

            avg_diversity = sum(diversities) / len(diversities) if diversities else 0.0
            rewards.append(avg_diversity)
            all_diversities.extend(diversities)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        return RewardOutput(
            rewards=rewards_tensor,
            metrics={
                f"{self.name}/mean": rewards_tensor.mean().item(),
                f"{self.name}/avg_pairwise_distance": (
                    sum(all_diversities) / len(all_diversities)
                    if all_diversities else 0
                ),
            },
        )

    def _get_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """Get embeddings for texts using the embedding model."""
        embeddings = []

        for text in texts:
            # Check cache
            cache_key = hash(text)
            if cache_key in self._embedding_cache:
                embeddings.append(self._embedding_cache[cache_key])
                continue

            # Compute embedding
            if self.embedding_model is not None and self.tokenizer is not None:
                emb = self._compute_embedding(text)
            else:
                # Fallback: simple hash-based embedding
                emb = self._simple_embedding(text)

            self._embedding_cache[cache_key] = emb
            embeddings.append(emb)

        return embeddings

    def _compute_embedding(self, text: str) -> torch.Tensor:
        """Compute embedding using the model."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.embedding_model(**inputs, output_hidden_states=True)
            # Use last token of last layer as embedding
            hidden_states = outputs.hidden_states[-1]
            mask = inputs["attention_mask"]
            last_idx = mask.sum(dim=1) - 1
            embedding = hidden_states[0, last_idx.item(), :]

        return embedding.cpu()

    def _simple_embedding(self, text: str, dim: int = 256) -> torch.Tensor:
        """Simple fallback embedding based on text hashing."""
        # Use multiple hash functions for diversity
        embeddings = []
        for seed in range(dim):
            h = hash(text + str(seed)) % (2**32)
            embeddings.append((h / (2**32)) * 2 - 1)  # Normalize to [-1, 1]

        return torch.tensor(embeddings, dtype=torch.float32)

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embedding_cache = {}


class GroupDiversityReward(BaseRewardComponent):
    """
    Diversity across GRPO group.

    Measures how different each trajectory's plans are from
    other trajectories in the same group.
    """

    def __init__(
        self,
        weight: float = 0.1,
        embedding_model=None,
        tokenizer=None,
        device: str = "cuda:0",
        **kwargs,
    ):
        super().__init__(weight=weight, name="group_diversity", **kwargs)
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.device = device

    def compute_for_group(
        self,
        group_actions: List[List[DebateAction]],
        **kwargs,
    ) -> RewardOutput:
        """
        Compute diversity rewards for a GRPO group.

        Args:
            group_actions: List of action sequences for K trajectories

        Returns:
            RewardOutput with per-trajectory diversity rewards
        """
        # Aggregate plans from each trajectory
        trajectory_plans = []
        for actions in group_actions:
            plans = " ".join([a.plan for a in actions if hasattr(a, 'plan') and a.plan])
            trajectory_plans.append(plans)

        # Get embeddings
        embeddings = self._get_embeddings(trajectory_plans)

        # Compute distance from each to all others
        rewards = []
        for i, emb in enumerate(embeddings):
            other_embs = embeddings[:i] + embeddings[i+1:]
            if other_embs:
                other_tensor = torch.stack(other_embs)
                similarities = F.cosine_similarity(emb.unsqueeze(0), other_tensor)
                avg_distance = 1 - similarities.mean().item()
            else:
                avg_distance = 0.0
            rewards.append(avg_distance)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        return RewardOutput(
            rewards=rewards_tensor,
            metrics={
                f"{self.name}/mean": rewards_tensor.mean().item(),
                f"{self.name}/std": rewards_tensor.std().item(),
            },
        )

    def _get_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """Get embeddings for texts."""
        embeddings = []

        for text in texts:
            if self.embedding_model is not None and self.tokenizer is not None:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.embedding_model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    embedding = hidden_states[0, -1, :]

                embeddings.append(embedding.cpu())
            else:
                # Fallback
                emb = torch.randn(256)
                emb = emb / emb.norm()
                embeddings.append(emb)

        return embeddings

    def compute(
        self,
        debate_results: List,
        **kwargs,
    ) -> RewardOutput:
        """Default compute - just returns zeros if not given group."""
        rewards = torch.zeros(len(debate_results))
        return RewardOutput(
            rewards=rewards,
            metrics={f"{self.name}/mean": 0.0},
        )
