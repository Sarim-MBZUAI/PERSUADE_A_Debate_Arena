"""
DPO Trainer — Direct Preference Optimization on Arena Debate Pairs

Stage 4: Polish training using preference pairs from arena outcomes.
Learns to prefer winning arguments over losing arguments on the same
topic/position/turn.
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
from ..data.arena_loader import ArenaDataLoader, _strip_think_tags
from ..losses import create_loss, LossOutput
from ..losses.dpo import DPOLoss
from ..losses.grpo import GRPOLoss
from ..utils.text import (
    get_system_prompt,
    build_user_prompt,
    pseudo_plan_from_argument,
)

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    chosen_model_filter: str = "kimi-k2"
    beta: float = 0.1
    label_smoothing: float = 0.0
    mode: str = "dpo"  # "dpo" or "filtered_grpo"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048
    include_plan: bool = True
    save_steps: int = 500
    eval_steps: int = 200
    logging_steps: int = 10
    bf16: bool = True


class PreferencePairDataset(Dataset):
    """
    Dataset of (chosen, rejected) argument pairs for DPO training.

    For each topic+position+turn, pairs winning arguments with losing
    arguments from arena debates.
    """

    def __init__(
        self,
        pairs: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2048,
        include_plan: bool = True,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.include_plan = include_plan

    @classmethod
    def from_arena_data(
        cls,
        arena: ArenaDataLoader,
        tokenizer: AutoTokenizer,
        config: DPOConfig,
        split: str = "train",
    ) -> "PreferencePairDataset":
        """Build preference pairs from arena data."""
        raw_pairs = arena.get_preference_pairs(
            chosen_model=config.chosen_model_filter,
            split=split,
        )

        logger.info(f"Built PreferencePairDataset with {len(raw_pairs)} pairs")
        return cls(
            raw_pairs, tokenizer, config.max_seq_length, config.include_plan,
        )

    def _build_prompt(self, topic: str, position: str, turn: int) -> str:
        """Build the shared prompt prefix for a preference pair."""
        system_prompt = get_system_prompt(position, include_plan=self.include_plan)
        user_prompt = build_user_prompt(
            topic=topic,
            position=position,
            turn=turn,
            debate_history=[],
            include_plan_instruction=self.include_plan,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _format_argument(self, argument: str) -> str:
        """Format argument with plan/debate tags."""
        if self.include_plan:
            plan = pseudo_plan_from_argument(argument)
            return f"<plan>\n{plan}\n</plan>\n<debate>\n{argument}\n</debate>"
        return argument

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        prompt = self._build_prompt(
            pair["topic"], pair["position"], pair["turn"],
        )

        chosen_text = prompt + self._format_argument(pair["chosen_argument"])
        rejected_text = prompt + self._format_argument(pair["rejected_argument"])

        # Tokenize
        chosen_enc = self.tokenizer(
            chosen_text, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )

        prompt_len = prompt_enc["input_ids"].shape[1]

        # Build masks (1 on target tokens, 0 on prompt)
        chosen_ids = chosen_enc["input_ids"].squeeze(0)
        rejected_ids = rejected_enc["input_ids"].squeeze(0)

        chosen_mask = torch.zeros_like(chosen_ids, dtype=torch.float)
        chosen_mask[prompt_len:] = 1.0

        rejected_mask = torch.zeros_like(rejected_ids, dtype=torch.float)
        rejected_mask[prompt_len:] = 1.0

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_mask": rejected_mask,
        }


def _dpo_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad batch to max length per field."""
    chosen_max = max(item["chosen_input_ids"].size(0) for item in batch)
    rejected_max = max(item["rejected_input_ids"].size(0) for item in batch)

    def pad_field(items, key, max_len, pad_value=0):
        tensors = []
        for item in items:
            t = item[key]
            pad_len = max_len - t.size(0)
            if pad_len > 0:
                t = torch.nn.functional.pad(t, (0, pad_len), value=pad_value)
            tensors.append(t)
        return torch.stack(tensors)

    return {
        "chosen_input_ids": pad_field(batch, "chosen_input_ids", chosen_max),
        "chosen_mask": pad_field(batch, "chosen_mask", chosen_max, 0.0),
        "rejected_input_ids": pad_field(batch, "rejected_input_ids", rejected_max),
        "rejected_mask": pad_field(batch, "rejected_mask", rejected_max, 0.0),
    }


class DPOTrainer:
    """
    DPO trainer for debate agents.

    Supports two modes:
    - "dpo": Standard DPO with preference pairs
    - "filtered_grpo": Treat (chosen, rejected) as 2-item GRPO group
      with advantages (+1, -1)
    """

    def __init__(
        self,
        config: DPOConfig,
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
        if ref_model is not None:
            self.ref_model = ref_model
        else:
            # Use the base model before LoRA as reference
            self.ref_model = self.model.base_model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Loss function
        if config.mode == "dpo":
            self.loss_fn = DPOLoss({
                "beta": config.beta,
                "label_smoothing": config.label_smoothing,
            })
        else:
            # filtered_grpo mode
            self.loss_fn = create_loss("grpo", {
                "beta": config.beta,
                "epsilon": 0.2,
            })

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

    def _get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
        with_grad: bool = True,
    ) -> torch.Tensor:
        """Compute per-token log probs for given input."""
        if with_grad:
            outputs = model(input_ids=input_ids)
        else:
            with torch.no_grad():
                outputs = model(input_ids=input_ids)

        logits = outputs.logits
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs_all = torch.log_softmax(shift_logits, dim=-1)
        log_probs = torch.gather(
            log_probs_all, dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # Adjust mask for the shift (remove first position)
        shifted_mask = mask[:, 1:]

        return log_probs * shifted_mask

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single DPO training step."""
        self.model.train()

        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_mask"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_mask"].to(self.device)

        if self.use_amp:
            with autocast("cuda"):
                # Policy log probs
                chosen_logps = self._get_log_probs(
                    self.model, chosen_ids, chosen_mask, with_grad=True,
                )
                rejected_logps = self._get_log_probs(
                    self.model, rejected_ids, rejected_mask, with_grad=True,
                )

                # Reference log probs
                ref_chosen_logps = self._get_log_probs(
                    self.ref_model, chosen_ids, chosen_mask, with_grad=False,
                )
                ref_rejected_logps = self._get_log_probs(
                    self.ref_model, rejected_ids, rejected_mask, with_grad=False,
                )

                if self.config.mode == "dpo":
                    loss_output = self.loss_fn.compute_loss(
                        log_probs=chosen_logps,
                        old_log_probs=rejected_logps,
                        ref_log_probs=ref_chosen_logps,
                        advantages=torch.zeros(chosen_ids.size(0), device=self.device),
                        mask=chosen_mask[:, 1:],
                        rejected_ref_log_probs=ref_rejected_logps,
                        rejected_mask=rejected_mask[:, 1:],
                    )
                else:
                    # filtered_grpo: stack chosen+rejected as 2-item group
                    all_logps = torch.cat([chosen_logps, rejected_logps], dim=0)
                    all_old_logps = all_logps.detach()
                    all_ref_logps = torch.cat(
                        [ref_chosen_logps, ref_rejected_logps], dim=0,
                    )
                    # Pad masks to same length
                    max_len = max(chosen_logps.size(1), rejected_logps.size(1))
                    c_mask = torch.nn.functional.pad(
                        chosen_mask[:, 1:], (0, max_len - chosen_mask[:, 1:].size(1)),
                    )
                    r_mask = torch.nn.functional.pad(
                        rejected_mask[:, 1:], (0, max_len - rejected_mask[:, 1:].size(1)),
                    )
                    all_masks = torch.cat([c_mask, r_mask], dim=0)
                    # Advantages: +1 for chosen, -1 for rejected
                    bs = chosen_ids.size(0)
                    advantages = torch.cat([
                        torch.ones(bs, device=self.device),
                        -torch.ones(bs, device=self.device),
                    ])

                    # Pad log probs to same length
                    all_logps_padded = torch.nn.functional.pad(
                        all_logps, (0, max_len - all_logps.size(1)),
                    )
                    all_old_logps_padded = torch.nn.functional.pad(
                        all_old_logps, (0, max_len - all_old_logps.size(1)),
                    )
                    all_ref_logps_padded = torch.nn.functional.pad(
                        all_ref_logps, (0, max_len - all_ref_logps.size(1)),
                    )

                    loss_output = self.loss_fn.compute_loss(
                        all_logps_padded, all_old_logps_padded,
                        all_ref_logps_padded, advantages, all_masks,
                    )

            loss = loss_output.loss / self.config.gradient_accumulation_steps
            loss.backward()
        else:
            # Same as above without autocast
            chosen_logps = self._get_log_probs(
                self.model, chosen_ids, chosen_mask, with_grad=True,
            )
            rejected_logps = self._get_log_probs(
                self.model, rejected_ids, rejected_mask, with_grad=True,
            )
            ref_chosen_logps = self._get_log_probs(
                self.ref_model, chosen_ids, chosen_mask, with_grad=False,
            )
            ref_rejected_logps = self._get_log_probs(
                self.ref_model, rejected_ids, rejected_mask, with_grad=False,
            )

            if self.config.mode == "dpo":
                loss_output = self.loss_fn.compute_loss(
                    log_probs=chosen_logps,
                    old_log_probs=rejected_logps,
                    ref_log_probs=ref_chosen_logps,
                    advantages=torch.zeros(chosen_ids.size(0), device=self.device),
                    mask=chosen_mask[:, 1:],
                    rejected_ref_log_probs=ref_rejected_logps,
                    rejected_mask=rejected_mask[:, 1:],
                )
            else:
                all_logps = torch.cat([chosen_logps, rejected_logps], dim=0)
                all_old_logps = all_logps.detach()
                all_ref_logps = torch.cat(
                    [ref_chosen_logps, ref_rejected_logps], dim=0,
                )
                max_len = max(chosen_logps.size(1), rejected_logps.size(1))
                c_mask = torch.nn.functional.pad(
                    chosen_mask[:, 1:], (0, max_len - chosen_mask[:, 1:].size(1)),
                )
                r_mask = torch.nn.functional.pad(
                    rejected_mask[:, 1:], (0, max_len - rejected_mask[:, 1:].size(1)),
                )
                all_masks = torch.cat([c_mask, r_mask], dim=0)
                bs = chosen_ids.size(0)
                advantages = torch.cat([
                    torch.ones(bs, device=self.device),
                    -torch.ones(bs, device=self.device),
                ])
                all_logps_padded = torch.nn.functional.pad(
                    all_logps, (0, max_len - all_logps.size(1)),
                )
                all_old_logps_padded = torch.nn.functional.pad(
                    all_old_logps, (0, max_len - all_old_logps.size(1)),
                )
                all_ref_logps_padded = torch.nn.functional.pad(
                    all_ref_logps, (0, max_len - all_ref_logps.size(1)),
                )
                loss_output = self.loss_fn.compute_loss(
                    all_logps_padded, all_old_logps_padded,
                    all_ref_logps_padded, advantages, all_masks,
                )

            loss = loss_output.loss / self.config.gradient_accumulation_steps
            loss.backward()

        return loss_output.metrics

    def train(
        self,
        arena: ArenaDataLoader,
        output_dir: str = "outputs/dpo",
    ) -> Dict[str, List[float]]:
        """Main DPO training loop."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build dataset
        dataset = PreferencePairDataset.from_arena_data(
            arena, self.tokenizer, self.config,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=_dpo_collate_fn,
            num_workers=0,
        )

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

        history = {"loss": [], "accuracy": []}
        global_step = 0

        for epoch in range(self.config.num_epochs):
            pbar = tqdm(dataloader, desc=f"DPO Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                metrics = self.training_step(batch)

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    loss_val = metrics.get("dpo/loss", metrics.get("grpo/total_loss", 0))
                    acc_val = metrics.get("dpo/accuracy", 0)
                    history["loss"].append(loss_val)
                    history["accuracy"].append(acc_val)

                    pbar.set_postfix({
                        "loss": f"{loss_val:.4f}",
                        "acc": f"{acc_val:.2%}" if acc_val else "N/A",
                    })

                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(
                            output_path / f"checkpoint-{global_step}"
                        )

        self.save_checkpoint(output_path / "final")
        return history

    def save_checkpoint(self, path: Path) -> None:
        """Save checkpoint (compatible with SPARTrainer format)."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "model")
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")

        state = {
            "paradigm": "dpo",
            "mode": self.config.mode,
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
