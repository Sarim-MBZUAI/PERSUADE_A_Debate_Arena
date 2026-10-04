"""
Hybrid Trainer — Online GRPO with Dataset Opponents

Stage 3: The trainable agent generates live responses while the opponent
replays stored arena responses. Combines benefits of online generation
(exploration, on-policy updates) with strong static opponents.
"""

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

from ..config import SPARConfig
from ..data.arena_loader import ArenaDataLoader
from ..debate.agents import TrainableAgent, FrozenAgent
from ..debate.dataset_agent import DatasetOpponentAgent
from ..debate.environment import DebateEnvironment, DebateResult
from ..debate.judge import JudgeInterface
from ..losses import create_loss, LossOutput
from ..rewards import HybridRewardAggregator, create_aggregator

logger = logging.getLogger(__name__)


@dataclass
class HybridTrainingConfig:
    """Configuration for hybrid training."""

    opponent_model_filter: Optional[str] = "kimi-k2"
    train_position: str = "random"  # "FOR", "AGAINST", or "random"
    num_generations: int = 4  # K trajectories per topic (GRPO group size)
    use_live_judge: bool = True
    num_epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    kl_coef: float = 0.1
    clip_epsilon: float = 0.2
    loss_type: str = "grpo"
    save_steps: int = 200
    eval_steps: int = 100
    logging_steps: int = 10
    bf16: bool = True


class HybridTrainer:
    """
    Online GRPO with dataset opponents.

    Generates live trajectories where:
    - Trainable agent (4B) generates plans + arguments in real-time
    - Opponent agent replays stored arena responses (e.g. kimi-k2)
    - Judge evaluates the live debate
    - Policy updated via GRPO with multi-component rewards
    """

    def __init__(
        self,
        config: HybridTrainingConfig,
        spar_config: SPARConfig,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        judge: JudgeInterface,
        arena: ArenaDataLoader,
        ref_model: Optional[AutoModelForCausalLM] = None,
        device: str = "cuda:0",
    ):
        self.config = config
        self.spar_config = spar_config
        self.device = device
        self.tokenizer = tokenizer
        self.judge = judge
        self.arena = arena

        # Setup model with LoRA
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=spar_config.model.lora_r,
            lora_alpha=spar_config.model.lora_alpha,
            lora_dropout=spar_config.model.lora_dropout,
            target_modules=spar_config.model.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, lora_config)
        self.model.to(device)

        # Reference model (frozen)
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.to(device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        # Loss function
        loss_config = {
            "beta": config.kl_coef,
            "epsilon": config.clip_epsilon,
            "epsilon_high": config.clip_epsilon,
        }
        self.loss_fn = create_loss(config.loss_type, loss_config)

        # Reward aggregator
        self.reward_aggregator = create_aggregator(
            config=spar_config.reward,
            embedding_model=model,
            tokenizer=tokenizer,
            device=device,
        )

        # Optimizer
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.95),
        )

        self.use_amp = config.bf16
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.global_step = 0
        self.run_name = f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def generate_hybrid_trajectories(
        self,
        topics: List[str],
        K: int = 4,
    ) -> Tuple[List[List[DebateResult]], List[List[Dict]]]:
        """
        Generate K trajectories per topic with dataset opponents.

        Returns:
            - group_results: [num_topics, K] debate results
            - group_actions: [num_topics, K] action data with log probs
        """
        group_results = []
        group_actions = []

        for topic in topics:
            topic_results = []
            topic_actions = []

            # Choose training position
            if self.config.train_position == "random":
                train_position = random.choice(["FOR", "AGAINST"])
            else:
                train_position = self.config.train_position

            opponent_position = "AGAINST" if train_position == "FOR" else "FOR"

            # Create agents
            trainable_agent = TrainableAgent(
                position=train_position,
                model=self.model,
                tokenizer=self.tokenizer,
                config=self.spar_config.debate,
            )

            dataset_opponent = DatasetOpponentAgent(
                position=opponent_position,
                arena=self.arena,
                config=self.spar_config.debate,
                model_filter=self.config.opponent_model_filter,
            )

            # Create environment
            if train_position == "FOR":
                env = DebateEnvironment(
                    for_agent=trainable_agent,
                    against_agent=dataset_opponent,
                    judge=self.judge,
                    config=self.spar_config.debate,
                    judge_config=self.spar_config.judge,
                )
            else:
                env = DebateEnvironment(
                    for_agent=dataset_opponent,
                    against_agent=trainable_agent,
                    judge=self.judge,
                    config=self.spar_config.debate,
                    judge_config=self.spar_config.judge,
                )

            for k in range(K):
                # Run debate
                result = env.run_debate(topic=topic, domain="")

                # Collect log probs for trainable agent
                train_actions = (
                    result.for_actions if train_position == "FOR"
                    else result.against_actions
                )

                action_data = {
                    "train_position": train_position,
                    "train": {
                        "actions": train_actions,
                        "log_probs": [],
                        "masks": [],
                    },
                }

                # Get log probs for each action
                for action in train_actions:
                    log_probs, _, mask = trainable_agent.get_log_probs(
                        action.raw_text if action.raw_text else action.format_with_plan(),
                        getattr(action, 'state_text', ''),
                    )
                    action_data["train"]["log_probs"].append(log_probs)
                    action_data["train"]["masks"].append(mask)

                topic_results.append(result)
                topic_actions.append(action_data)

            group_results.append(topic_results)
            group_actions.append(topic_actions)

        return group_results, group_actions

    def training_step(self, topics: List[str]) -> Dict[str, float]:
        """Single training step on a batch of topics."""
        self.model.train()

        # Step 1: Generate trajectories
        group_results, group_actions = self.generate_hybrid_trajectories(
            topics, K=self.config.num_generations,
        )

        # Step 2: Compute rewards and advantages
        all_rewards = []
        all_advantages = []

        for topic_results, topic_actions in zip(group_results, group_actions):
            if not topic_actions:
                continue

            train_position = topic_actions[0]["train_position"]

            output = self.reward_aggregator.compute(
                debate_results=topic_results,
                for_actions=[a["train"]["actions"] for a in topic_actions],
                position=train_position,
            )

            all_rewards.extend(output.rewards.tolist())
            all_advantages.extend(output.advantages.tolist())

        advantages = torch.tensor(all_advantages, device=self.device)

        # Step 3: Collect log probs and compute loss
        self.optimizer.zero_grad()

        all_old_log_probs = []
        all_ref_log_probs = []
        all_masks = []
        expanded_advantages = []
        action_texts = []

        adv_idx = 0
        for topic_actions in group_actions:
            for action_data in topic_actions:
                traj_advantage = advantages[adv_idx].item()
                adv_idx += 1
                train_data = action_data["train"]
                train_position = action_data["train_position"]

                for i, (lp, mask) in enumerate(zip(
                    train_data["log_probs"], train_data["masks"]
                )):
                    all_old_log_probs.append(lp.detach())
                    all_ref_log_probs.append(lp.detach())
                    all_masks.append(mask)
                    expanded_advantages.append(traj_advantage)

                    action = train_data["actions"][i]
                    action_texts.append((
                        action.raw_text if action.raw_text else action.format_with_plan(),
                        getattr(action, 'state_text', ''),
                        train_position,
                    ))

        advantages = torch.tensor(expanded_advantages, device=self.device)

        # Re-compute log probs with gradients
        trainable_for = TrainableAgent(
            position="FOR", model=self.model,
            tokenizer=self.tokenizer, config=self.spar_config.debate,
        )
        trainable_against = TrainableAgent(
            position="AGAINST", model=self.model,
            tokenizer=self.tokenizer, config=self.spar_config.debate,
        )

        all_log_probs = []
        for raw_text, state_text, position in action_texts:
            agent = trainable_for if position == "FOR" else trainable_against
            lp, _, _ = agent.get_log_probs(raw_text, state_text, with_grad=True)
            all_log_probs.append(lp)

        # Pad sequences
        max_len = max(lp.size(-1) for lp in all_log_probs)

        def pad_list(tensors):
            padded = []
            for t in tensors:
                pad_len = max_len - t.size(-1)
                if pad_len > 0:
                    t = torch.nn.functional.pad(t, (0, pad_len), value=0)
                padded.append(t)
            return torch.stack(padded)

        log_probs = pad_list(all_log_probs)
        old_log_probs = pad_list(all_old_log_probs)
        ref_log_probs = pad_list(all_ref_log_probs)
        masks = pad_list(all_masks)

        if self.use_amp:
            with autocast("cuda"):
                loss_output = self.loss_fn.compute_loss(
                    log_probs, old_log_probs, ref_log_probs,
                    advantages, masks,
                )
            self.scaler.scale(loss_output.loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss_output = self.loss_fn.compute_loss(
                log_probs, old_log_probs, ref_log_probs,
                advantages, masks,
            )
            loss_output.loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.global_step += 1

        # Compute win rate
        wins = sum(
            1 for topic_results in group_results
            for r in topic_results
            if r.winner == "FOR"
        )
        total = sum(len(tr) for tr in group_results)

        metrics = {
            "loss": loss_output.loss.item(),
            "policy_loss": loss_output.policy_loss.item(),
            "kl_loss": loss_output.kl_loss.item(),
            "win_rate": wins / total if total > 0 else 0.0,
            "reward_mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        }

        return metrics

    def train(
        self,
        train_topics: List[str],
        eval_topics: Optional[List[str]] = None,
        output_dir: str = "outputs/hybrid",
    ) -> Dict[str, List[float]]:
        """Main training loop."""
        output_path = Path(output_dir) / self.run_name
        output_path.mkdir(parents=True, exist_ok=True)

        steps_per_epoch = max(1, len(train_topics) // self.config.batch_size)
        total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        history = {"loss": [], "win_rate": []}

        for epoch in range(self.config.num_epochs):
            shuffled = train_topics.copy()
            random.shuffle(shuffled)

            pbar = tqdm(
                range(steps_per_epoch),
                desc=f"Hybrid Epoch {epoch + 1}/{self.config.num_epochs}",
            )

            for step_idx in pbar:
                batch_start = step_idx * self.config.batch_size
                batch_topics = shuffled[batch_start:batch_start + self.config.batch_size]

                metrics = self.training_step(batch_topics)
                scheduler.step()

                history["loss"].append(metrics["loss"])
                history["win_rate"].append(metrics["win_rate"])

                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "win": f"{metrics['win_rate']:.2%}",
                })

                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(
                        output_path / f"checkpoint-{self.global_step}"
                    )

        self.save_checkpoint(output_path / "final")
        return history

    def save_checkpoint(self, path: Path) -> None:
        """Save checkpoint (compatible with SPARTrainer format)."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "model")
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")

        state = {
            "paradigm": "hybrid",
            "global_step": self.global_step,
            "config": asdict(self.config),
        }
        with open(path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def load_checkpoint(self, path: Path) -> None:
        """Load a checkpoint."""
        self.model = PeftModel.from_pretrained(
            self.model.base_model,
            path / "model",
        )
        self.model.to(self.device)

        optimizer_path = path / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=self.device)
            )

        state_path = path / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
