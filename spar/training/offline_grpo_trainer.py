"""
Offline GRPO Trainer

Stage 2: Group Relative Policy Optimization on pre-recorded arena debates.
Instead of generating live trajectories, uses stored arena debates as the
GRPO group and computes advantages from arena outcomes + text-based rewards.
"""

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

from ..config import SPARConfig
from ..data.arena_loader import ArenaDataLoader, ArenaDebate, _strip_think_tags
from ..debate.agents import LLMDebateAgent
from ..losses import create_loss, BasePolicyLoss, LossOutput
from ..rewards.format import FormatReward
from ..rewards.diversity import PlanDiversityReward
from ..utils.text import (
    get_system_prompt,
    build_user_prompt,
    pseudo_plan_from_argument,
)

logger = logging.getLogger(__name__)


@dataclass
class OfflineGRPOConfig:
    """Configuration for offline GRPO training."""

    min_debates_per_topic: int = 4
    position_filter: Optional[str] = None  # "FOR", "AGAINST", or None for both
    model_filter: Optional[str] = None
    use_arena_outcome: bool = True
    num_epochs: int = 3
    batch_size: int = 2  # Topics per batch
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    kl_coef: float = 0.1
    clip_epsilon: float = 0.2
    loss_type: str = "grpo"
    for_win_bonus: float = 1.5
    gradient_accumulation_steps: int = 4
    save_steps: int = 200
    eval_steps: int = 100
    logging_steps: int = 10
    bf16: bool = True


class OfflineGRPOTrainer:
    """
    Offline GRPO trainer using pre-recorded arena debates.

    For each topic:
    1. Group K debates as the GRPO "group"
    2. Compute rewards from arena outcomes + text-based rewards
    3. Compute group-relative advantages
    4. Get log probs of arena arguments from the current policy
    5. Update policy via GRPO/SPAR loss
    """

    def __init__(
        self,
        config: OfflineGRPOConfig,
        spar_config: SPARConfig,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        ref_model: Optional[AutoModelForCausalLM] = None,
        device: str = "cuda:0",
    ):
        self.config = config
        self.spar_config = spar_config
        self.device = device
        self.tokenizer = tokenizer

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

        # Text-based reward components
        self.format_reward = FormatReward(weight=0.05)
        self.diversity_reward = PlanDiversityReward(weight=0.1)

        # Trainable agent for computing log probs
        self.agent = LLMDebateAgent(
            position="FOR",  # Position will be overridden per-example
            model=self.model,
            tokenizer=self.tokenizer,
            config=spar_config.debate,
            trainable=True,
        )

        # Reference agent for computing ref log probs
        self.ref_agent = None
        if self.ref_model is not None:
            self.ref_agent = LLMDebateAgent(
                position="FOR",
                model=self.ref_model,
                tokenizer=self.tokenizer,
                config=spar_config.debate,
                trainable=False,
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

    def _compute_offline_rewards(
        self,
        debates: List[ArenaDebate],
        position: str,
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        Compute rewards for a group of debates from arena outcomes.

        Args:
            debates: List of ArenaDebate for one topic
            position: Position to compute rewards for

        Returns:
            Tuple of (rewards tensor [num_debates], reward_breakdown dict)
        """
        rewards = []
        breakdown = {
            "outcome_reward": [],
            "format_reward": [],
            "diversity_reward": [],
            "total_reward": [],
        }

        for debate in debates:
            reward = 0.0
            outcome_r = 0.0
            format_r = 0.0
            diversity_r = 0.0

            # Outcome reward from arena winner
            if self.config.use_arena_outcome:
                if debate.winner_position == position:
                    outcome_r = 1.0
                    if position == "FOR":
                        outcome_r *= self.config.for_win_bonus
                else:
                    outcome_r = -1.0
                reward += outcome_r

            # Format reward (text-based)
            result = debate.to_debate_result()
            actions = result.for_actions if position == "FOR" else result.against_actions
            fmt_output = self.format_reward.compute(
                debate_results=[result],
                for_actions=[actions],
            )
            format_r = self.format_reward.weight * fmt_output.rewards[0].item()
            reward += format_r

            # Diversity reward
            div_output = self.diversity_reward.compute(
                debate_results=[result],
                for_actions=[actions],
            )
            diversity_r = self.diversity_reward.weight * div_output.rewards[0].item()
            reward += diversity_r

            rewards.append(reward)
            breakdown["outcome_reward"].append(outcome_r)
            breakdown["format_reward"].append(format_r)
            breakdown["diversity_reward"].append(diversity_r)
            breakdown["total_reward"].append(reward)

        return torch.tensor(rewards, dtype=torch.float32, device=self.device), breakdown

    def _build_context(
        self,
        debate: ArenaDebate,
        turn_entry: "ArenaTurn",
        history_before: List[Dict[str, Any]],
    ) -> str:
        """Build prompt context for computing log probs on an arena argument."""
        system_prompt = get_system_prompt(
            turn_entry.position,
            include_plan=True,
        )
        user_prompt = build_user_prompt(
            topic=debate.topic,
            position=turn_entry.position,
            turn=turn_entry.turn,
            debate_history=history_before,
            include_plan_instruction=True,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def training_step(
        self,
        topic_debates: Dict[str, List[ArenaDebate]],
    ) -> Dict[str, float]:
        """
        Single training step on a batch of topic groups.

        For each topic:
        1. Take K debates as GRPO group
        2. Compute rewards and advantages
        3. Get log probs and update policy
        """
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_items = 0
        all_metrics = {}

        # Comprehensive tracking
        step_reward_breakdowns = {
            "outcome_reward": [], "format_reward": [],
            "diversity_reward": [], "total_reward": [],
        }
        step_advantages = []
        step_completion_lengths = []
        step_context_lengths = []
        step_kl_per_token = []
        step_num_debates = 0
        step_num_turns = 0
        step_positions = []

        for topic, debates in topic_debates.items():
            K = min(len(debates), self.config.min_debates_per_topic)
            group = random.sample(debates, K) if len(debates) > K else debates
            step_num_debates += len(group)

            # Pick a training position
            position = self.config.position_filter or random.choice(["FOR", "AGAINST"])
            step_positions.append(position)

            # Compute rewards and advantages (now returns breakdown)
            rewards, reward_breakdown = self._compute_offline_rewards(group, position)
            advantages = rewards - rewards.mean()
            if rewards.std() > 1e-8:
                advantages = advantages / (rewards.std() + 1e-8)

            # Track reward breakdown
            for k in step_reward_breakdowns:
                step_reward_breakdowns[k].extend(reward_breakdown[k])
            step_advantages.extend(advantages.tolist())

            # Process each debate's turns
            for debate_idx, debate in enumerate(group):
                turns = (
                    debate.for_turns if position == "FOR" else debate.against_turns
                )
                debate_advantage = advantages[debate_idx]

                # Build history incrementally for each turn
                history_so_far: List[Dict[str, Any]] = []
                all_turns_sorted = sorted(
                    debate.turns, key=lambda t: (t.turn, t.position != "FOR")
                )

                for turn_entry in all_turns_sorted:
                    if turn_entry.position == position:
                        step_num_turns += 1
                        argument = _strip_think_tags(turn_entry.argument)
                        plan = pseudo_plan_from_argument(argument)
                        target_text = f"<plan>\n{plan}\n</plan>\n<debate>\n{argument}\n</debate>"

                        context = self._build_context(
                            debate, turn_entry, history_so_far.copy()
                        )

                        # Get log probs WITH gradients
                        log_probs, _, mask = self.agent.get_log_probs(
                            target_text, context, with_grad=True
                        )

                        # Get old log probs (no grad)
                        old_log_probs, _, _ = self.agent.get_log_probs(
                            target_text, context, with_grad=False
                        )

                        # Get ref log probs
                        if self.ref_agent is not None:
                            ref_log_probs, _, _ = self.ref_agent.get_log_probs(
                                target_text, context, with_grad=False
                            )
                        else:
                            ref_log_probs = old_log_probs.detach()

                        # Ensure shapes match
                        min_len = min(
                            log_probs.size(-1),
                            old_log_probs.size(-1),
                            ref_log_probs.size(-1),
                            mask.size(-1),
                        )
                        log_probs = log_probs[..., :min_len].unsqueeze(0)
                        old_log_probs = old_log_probs[..., :min_len].unsqueeze(0)
                        ref_log_probs = ref_log_probs[..., :min_len].unsqueeze(0)
                        mask = mask[..., :min_len].unsqueeze(0)
                        adv = debate_advantage.unsqueeze(0)

                        # Track token lengths
                        num_completion_tokens = mask.sum().item()
                        step_completion_lengths.append(num_completion_tokens)

                        # Track context length (prompt + completion)
                        context_tokens = len(self.tokenizer.encode(context, add_special_tokens=False))
                        step_context_lengths.append(context_tokens)

                        # Compute per-token KL before loss
                        with torch.no_grad():
                            kl_per_token = (log_probs - ref_log_probs) * mask
                            kl_mean = kl_per_token.sum() / mask.sum()
                            step_kl_per_token.append(kl_mean.item())

                        loss_output = self.loss_fn.compute_loss(
                            log_probs, old_log_probs, ref_log_probs,
                            adv, mask,
                        )

                        scaled_loss = (
                            loss_output.loss / self.config.gradient_accumulation_steps
                        )
                        scaled_loss.backward()

                        total_loss += loss_output.loss.item()
                        total_items += 1

                        for k, v in loss_output.metrics.items():
                            all_metrics.setdefault(k, []).append(v)

                    # Update history
                    history_so_far.append({
                        "turn": turn_entry.turn,
                        "agent": turn_entry.position,
                        "argument": _strip_think_tags(turn_entry.argument),
                    })

        # Compute grad norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        # Build comprehensive metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
        avg_metrics["loss"] = total_loss / max(total_items, 1)

        # Reward breakdowns
        for k, v in step_reward_breakdowns.items():
            if v:
                avg_metrics[f"reward/{k}_mean"] = sum(v) / len(v)
                avg_metrics[f"reward/{k}_std"] = (
                    (sum((x - sum(v)/len(v))**2 for x in v) / len(v)) ** 0.5
                    if len(v) > 1 else 0.0
                )

        # Advantage stats
        if step_advantages:
            avg_metrics["advantage/mean"] = sum(step_advantages) / len(step_advantages)
            adv_mean = avg_metrics["advantage/mean"]
            avg_metrics["advantage/std"] = (
                (sum((a - adv_mean)**2 for a in step_advantages) / len(step_advantages)) ** 0.5
                if len(step_advantages) > 1 else 0.0
            )
            avg_metrics["advantage/min"] = min(step_advantages)
            avg_metrics["advantage/max"] = max(step_advantages)

        # Completion length stats
        if step_completion_lengths:
            avg_metrics["tokens/completion_mean"] = sum(step_completion_lengths) / len(step_completion_lengths)
            avg_metrics["tokens/completion_min"] = min(step_completion_lengths)
            avg_metrics["tokens/completion_max"] = max(step_completion_lengths)

        # Context length stats
        if step_context_lengths:
            avg_metrics["tokens/context_mean"] = sum(step_context_lengths) / len(step_context_lengths)
            avg_metrics["tokens/context_max"] = max(step_context_lengths)

        # KL stats
        if step_kl_per_token:
            avg_metrics["kl/per_token_mean"] = sum(step_kl_per_token) / len(step_kl_per_token)

        # Training stats
        avg_metrics["train/grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        avg_metrics["train/num_debates"] = step_num_debates
        avg_metrics["train/num_turns"] = step_num_turns
        avg_metrics["train/num_topics"] = len(topic_debates)

        return avg_metrics

    def train(
        self,
        arena: ArenaDataLoader,
        output_dir: str = "outputs/offline_grpo",
    ) -> Dict[str, List[float]]:
        """
        Main training loop over grouped arena topics.

        Args:
            arena: ArenaDataLoader with loaded data
            output_dir: Output directory

        Returns:
            Training history
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Setup metrics log file
        metrics_log_path = output_path / "metrics.jsonl"

        # Group debates by topic
        grouped = arena.get_debates_grouped_by_topic(
            split="train",
            min_per_topic=self.config.min_debates_per_topic,
            model_filter=self.config.model_filter,
        )
        topics = list(grouped.keys())
        logger.info(f"Offline GRPO: {len(topics)} topics with >= {self.config.min_debates_per_topic} debates")

        # Setup scheduler
        steps_per_epoch = max(1, len(topics) // self.config.batch_size)
        total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Log training configuration
        print(f"\n{'='*70}")
        print(f"  Total topics: {len(topics)}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Debates per topic (K): {self.config.min_debates_per_topic}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  KL coefficient (beta): {self.config.kl_coef}")
        print(f"  Clip epsilon: {self.config.clip_epsilon}")
        print(f"  Max grad norm: {self.config.max_grad_norm}")
        print(f"  Metrics log: {metrics_log_path}")
        print(f"{'='*70}\n")

        history = {"loss": []}
        global_step = 0

        for epoch in range(self.config.num_epochs):
            random.shuffle(topics)
            epoch_losses = []

            pbar = tqdm(
                range(steps_per_epoch),
                desc=f"Offline GRPO Epoch {epoch + 1}/{self.config.num_epochs}",
            )

            for step_idx in pbar:
                batch_start = step_idx * self.config.batch_size
                batch_topics = topics[batch_start:batch_start + self.config.batch_size]
                topic_debates = {t: grouped[t] for t in batch_topics if t in grouped}

                if not topic_debates:
                    continue

                metrics = self.training_step(topic_debates)
                scheduler.step()
                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                metrics["train/learning_rate"] = current_lr
                metrics["train/global_step"] = global_step
                metrics["train/epoch"] = epoch + 1

                history["loss"].append(metrics.get("loss", 0.0))
                epoch_losses.append(metrics.get("loss", 0.0))

                # Compact progress bar
                pbar.set_postfix({
                    "loss": f"{metrics.get('loss', 0):.4f}",
                    "kl": f"{metrics.get('kl/per_token_mean', 0):.4f}",
                    "rew": f"{metrics.get('reward/total_reward_mean', 0):.3f}",
                    "gnorm": f"{metrics.get('train/grad_norm', 0):.2f}",
                    "comp_tok": f"{metrics.get('tokens/completion_mean', 0):.0f}",
                })

                # Detailed logging every N steps
                if global_step % self.config.logging_steps == 0:
                    print(f"\n--- Step {global_step}/{total_steps} (Epoch {epoch+1}) ---")
                    print(f"  Loss:              {metrics.get('loss', 0):.6f}")
                    print(f"  Learning Rate:     {current_lr:.2e}")
                    print(f"  Grad Norm:         {metrics.get('train/grad_norm', 0):.4f}")
                    print(f"  -- Rewards --")
                    print(f"    Outcome (mean):  {metrics.get('reward/outcome_reward_mean', 0):.4f}")
                    print(f"    Format  (mean):  {metrics.get('reward/format_reward_mean', 0):.4f}")
                    print(f"    Diversity(mean): {metrics.get('reward/diversity_reward_mean', 0):.4f}")
                    print(f"    Total   (mean):  {metrics.get('reward/total_reward_mean', 0):.4f} +/- {metrics.get('reward/total_reward_std', 0):.4f}")
                    print(f"  -- Advantages --")
                    print(f"    Mean: {metrics.get('advantage/mean', 0):.4f}  Std: {metrics.get('advantage/std', 0):.4f}")
                    print(f"    Min:  {metrics.get('advantage/min', 0):.4f}  Max: {metrics.get('advantage/max', 0):.4f}")
                    print(f"  -- KL & Policy --")
                    print(f"    KL per token:    {metrics.get('kl/per_token_mean', 0):.6f}")
                    print(f"    GRPO approx KL:  {metrics.get('grpo/approx_kl', 0):.6f}")
                    print(f"    GRPO clip frac:  {metrics.get('grpo/clip_fraction', 0):.4f}")
                    print(f"    GRPO ratio mean: {metrics.get('grpo/ratio_mean', 0):.4f}")
                    print(f"  -- Tokens --")
                    print(f"    Completion: mean={metrics.get('tokens/completion_mean', 0):.0f}  min={metrics.get('tokens/completion_min', 0):.0f}  max={metrics.get('tokens/completion_max', 0):.0f}")
                    print(f"    Context:    mean={metrics.get('tokens/context_mean', 0):.0f}  max={metrics.get('tokens/context_max', 0):.0f}")
                    print(f"  -- Step Info --")
                    print(f"    Topics: {metrics.get('train/num_topics', 0):.0f}  Debates: {metrics.get('train/num_debates', 0):.0f}  Turns: {metrics.get('train/num_turns', 0):.0f}")

                # Write metrics to JSONL (every step)
                with open(metrics_log_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(
                        output_path / f"checkpoint-{global_step}"
                    )

            # End of epoch summary
            if epoch_losses:
                print(f"\n{'='*70}")
                print(f"  Epoch {epoch+1}/{self.config.num_epochs} Summary")
                print(f"  Mean Loss: {sum(epoch_losses)/len(epoch_losses):.6f}")
                print(f"  Min Loss:  {min(epoch_losses):.6f}")
                print(f"  Max Loss:  {max(epoch_losses):.6f}")
                print(f"{'='*70}\n")

        self.save_checkpoint(output_path / "final")
        return history

    def save_checkpoint(self, path: Path) -> None:
        """Save checkpoint (compatible with SPARTrainer format)."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "model")
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")

        state = {
            "paradigm": "offline_grpo",
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
