"""
SFT Trainer — Supervised Fine-Tuning on Expert Debate Demonstrations

Stage 1 of the multi-paradigm training pipeline.
Trains on winning debate turns from expert models (e.g. kimi-k2).
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

from ..config import SPARConfig
from ..data.arena_loader import ArenaDataLoader, ArenaDebate, ArenaTurn, _strip_think_tags
from ..utils.text import (
    get_system_prompt,
    build_user_prompt,
    pseudo_plan_from_argument,
)

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    expert_models: List[str] = field(default_factory=lambda: ["kimi-k2"])
    include_kimi_beaters: bool = True
    min_argument_length: int = 100
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 4096
    include_plan: bool = True
    save_steps: int = 500
    eval_steps: int = 200
    logging_steps: int = 10
    bf16: bool = True
    full_finetune: bool = False  # If True, skip LoRA and train all params


class DebateSFTDataset(Dataset):
    """
    Dataset of per-turn SFT examples from arena debate demonstrations.

    Each example: (system_prompt + debate_history_up_to_turn, target_argument)
    Target is formatted as <plan>...</plan><debate>...</debate>.
    Labels are -100 on prompt tokens and actual ids on target tokens.
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2048,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    @classmethod
    def from_arena_data(
        cls,
        arena: ArenaDataLoader,
        tokenizer: AutoTokenizer,
        config: SFTConfig,
        split: str = "train",
    ) -> "DebateSFTDataset":
        """
        Build SFT dataset from arena data.

        Sources:
          (a) All expert_models winning debate turns
          (b) Turns from models that beat an expert (if include_kimi_beaters)
          Filter: argument length >= min_argument_length, no mid-sentence truncation
        """
        demos = arena.get_expert_demonstrations(
            expert_models=config.expert_models,
            include_beaters=config.include_kimi_beaters,
            split=split,
        )

        examples = []
        for debate in demos:
            # Determine which side's turns to learn from
            # Winner's turns are the expert demonstrations
            winner_position = debate.winner_position
            winner_turns = (
                debate.for_turns if winner_position == "FOR" else debate.against_turns
            )
            all_turns = debate.turns

            # Build examples per turn
            history_so_far: List[Dict[str, Any]] = []
            for turn_entry in sorted(all_turns, key=lambda t: (t.turn, t.position != "FOR")):
                # If this is a winner's turn, create a training example
                if turn_entry.position == winner_position:
                    argument = _strip_think_tags(turn_entry.argument)

                    # Filter: minimum length
                    if len(argument) < config.min_argument_length:
                        continue

                    # Filter: no mid-sentence truncation (must end with sentence-ending punct)
                    if argument and argument[-1] not in ".!?\"')":
                        continue

                    # Build the target with plan+debate tags
                    plan = pseudo_plan_from_argument(argument)
                    if config.include_plan:
                        target = f"<plan>\n{plan}\n</plan>\n<debate>\n{argument}\n</debate>"
                    else:
                        target = argument

                    # Build the prompt (system + user prompt with history)
                    system_prompt = get_system_prompt(
                        winner_position,
                        include_plan=config.include_plan,
                    )
                    user_prompt = build_user_prompt(
                        topic=debate.topic,
                        position=winner_position,
                        turn=turn_entry.turn,
                        debate_history=history_so_far.copy(),
                        include_plan_instruction=config.include_plan,
                    )

                    examples.append({
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "target": target,
                        "topic": debate.topic,
                        "position": winner_position,
                        "turn": turn_entry.turn,
                        "model": turn_entry.model,
                    })

                # Always update history
                history_so_far.append({
                    "turn": turn_entry.turn,
                    "agent": turn_entry.position,
                    "argument": _strip_think_tags(turn_entry.argument),
                })

        logger.info(f"Built SFT dataset with {len(examples)} examples from {len(demos)} debates")
        return cls(examples, tokenizer, config.max_seq_length)

    def _tokenize_example(self, ex: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenize a single example. Returns None if no target tokens survive truncation."""
        messages = [
            {"role": "system", "content": ex["system_prompt"]},
            {"role": "user", "content": ex["user_prompt"]},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        target_text = ex["target"]
        full_text = prompt_text + target_text

        # Tokenize
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = full_encoding["input_ids"].squeeze(0)
        attention_mask = full_encoding["attention_mask"].squeeze(0)
        prompt_len = prompt_encoding["input_ids"].shape[1]

        # If prompt fills the entire sequence, no target tokens survive
        if prompt_len >= input_ids.shape[0]:
            return None

        # Labels: -100 on prompt tokens, actual ids on target
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def filter_valid_examples(self) -> int:
        """Pre-filter examples that would produce all-masked labels. Returns count removed."""
        valid = []
        removed = 0
        for ex in self.examples:
            result = self._tokenize_example(ex)
            if result is not None:
                valid.append(ex)
            else:
                removed += 1
        self.examples = valid
        if removed > 0:
            logger.info(f"Filtered {removed} examples with prompts exceeding max_seq_length")
        return removed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        result = self._tokenize_example(ex)
        if result is None:
            # Shouldn't happen after filtering, but safety fallback:
            # return next valid example
            for offset in range(1, len(self.examples)):
                result = self._tokenize_example(self.examples[(idx + offset) % len(self.examples)])
                if result is not None:
                    return result
        return result


def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad batch to max length."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(
            torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=0)
        )
        attention_mask.append(
            torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0)
        )
        labels.append(
            torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100)
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for debate agents.

    Trains on expert demonstrations using standard cross-entropy loss
    with LoRA, following the same patterns as SPARTrainer for
    checkpoint compatibility.
    """

    def __init__(
        self,
        config: SFTConfig,
        spar_config: SPARConfig,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda:0",
    ):
        self.config = config
        self.spar_config = spar_config
        self.device = device
        self.tokenizer = tokenizer
        self.full_finetune = config.full_finetune

        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        if self.full_finetune:
            # Full fine-tuning: train all parameters
            self.model = model.to(device)
            for param in self.model.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Full fine-tuning: {trainable:,} / {total:,} params trainable")
        else:
            # LoRA fine-tuning (same pattern as SPARTrainer._setup_lora)
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

    def train(
        self,
        dataset: DebateSFTDataset,
        eval_dataset: Optional[DebateSFTDataset] = None,
        output_dir: str = "outputs/sft",
    ) -> Dict[str, List[float]]:
        """
        Train on SFT dataset.

        Args:
            dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            output_dir: Directory for checkpoints

        Returns:
            Training history dict
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            num_workers=0,
        )

        # Scheduler
        total_steps = (
            len(dataloader) * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        history = {"loss": [], "eval_perplexity": []}
        global_step = 0
        best_eval_ppl = float("inf")

        self.model.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # No autocast needed — model is already in bf16
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps

                # NaN detection: skip bad batches
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        f"NaN/Inf loss at batch {batch_idx}, skipping. "
                        f"Labels non-masked: {(batch['labels'] != -100).sum().item()}"
                    )
                    self.optimizer.zero_grad()
                    continue

                loss.backward()

                epoch_loss += outputs.loss.item()
                num_batches += 1

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    avg_loss = epoch_loss / max(num_batches, 1)
                    history["loss"].append(avg_loss)

                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})

                    if global_step % self.config.logging_steps == 0:
                        logger.info(
                            f"Step {global_step}: loss={avg_loss:.4f}, "
                            f"lr={scheduler.get_last_lr()[0]:.2e}"
                        )

                    # Evaluation
                    if eval_dataset and global_step % self.config.eval_steps == 0:
                        eval_ppl = self._evaluate(eval_dataset)
                        history["eval_perplexity"].append(eval_ppl)
                        logger.info(f"Step {global_step}: eval_perplexity={eval_ppl:.2f}")

                        if eval_ppl < best_eval_ppl:
                            best_eval_ppl = eval_ppl
                            self.save_checkpoint(output_path / "best")

                        self.model.train()

                    # Checkpoint
                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(
                            output_path / f"checkpoint-{global_step}"
                        )

        # Final save
        self.save_checkpoint(output_path / "final")

        return history

    @torch.no_grad()
    def _evaluate(self, dataset: DebateSFTDataset) -> float:
        """Compute perplexity on eval dataset."""
        self.model.eval()

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
        )

        total_loss = 0.0
        total_tokens = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            # Count non-ignored tokens
            valid_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity

    def save_checkpoint(self, path: Path) -> None:
        """Save checkpoint (compatible with SPARTrainer format)."""
        path.mkdir(parents=True, exist_ok=True)

        if self.full_finetune:
            # Save full model weights
            self.model.save_pretrained(path / "model")
            self.tokenizer.save_pretrained(path / "model")
        else:
            # Save LoRA weights
            self.model.save_pretrained(path / "model")

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")

        # Save training state
        state = {
            "paradigm": "sft",
            "full_finetune": self.full_finetune,
            "config": asdict(self.config),
        }
        with open(path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def load_checkpoint(self, path: Path) -> None:
        """Load a checkpoint."""
        state_path = path / "training_state.json"
        is_full = self.full_finetune
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            is_full = state.get("full_finetune", False)

        if is_full:
            self.model = AutoModelForCausalLM.from_pretrained(
                path / "model",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device)
        else:
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
