#!/usr/bin/env python3
"""
Offline GRPO Training on Arena Debates

Stage 2: Group Relative Policy Optimization using pre-recorded arena debates
as the GRPO group, with rewards derived from arena outcomes.

Usage:
    python scripts/train_offline_grpo.py \
        --config configs/offline_grpo_arena.yaml \
        --model Qwen/Qwen3-4B \
        --resume outputs/sft/final \
        --output-dir outputs/offline_grpo
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spar.config import SPARConfig, load_config
from spar.data.arena_loader import ArenaDataLoader
from spar.training.offline_grpo_trainer import OfflineGRPOConfig, OfflineGRPOTrainer
from spar.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Offline GRPO training on arena debates")

    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--model", type=str, default=None, help="Model override")
    parser.add_argument("--arena-data-dir", type=str, default="data/debate-arena")
    parser.add_argument(
        "--loss-type", type=str, default=None,
        choices=["grpo"],
        help="Loss function type",
    )
    parser.add_argument("--model-filter", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/offline_grpo")
    parser.add_argument("--resume", type=str, default=None,
                        help="SFT checkpoint to load as base model (full FT or LoRA)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load config
    print(f"Loading config from {args.config}")
    spar_config = load_config(args.config)

    if args.model:
        spar_config.model.name_or_path = args.model

    # Build offline GRPO config
    offline_config = OfflineGRPOConfig()
    if args.loss_type:
        offline_config.loss_type = args.loss_type
    if args.model_filter:
        offline_config.model_filter = args.model_filter
    if args.epochs:
        offline_config.num_epochs = args.epochs
    if args.batch_size:
        offline_config.batch_size = args.batch_size
    if args.lr:
        offline_config.learning_rate = args.lr

    # Load arena data
    print(f"\nLoading arena data from {args.arena_data_dir}")
    arena = ArenaDataLoader(data_dir=args.arena_data_dir)
    stats = arena.stats()
    print(f"  Train debates: {stats['train_debates']}")

    # Check grouped topics
    grouped = arena.get_debates_grouped_by_topic(
        "train", min_per_topic=offline_config.min_debates_per_topic,
        model_filter=offline_config.model_filter,
    )
    print(f"  Topics with >= {offline_config.min_debates_per_topic} debates: {len(grouped)}")

    # Determine model source: SFT checkpoint or HuggingFace
    model_path = spar_config.model.name_or_path
    resume_path = Path(args.resume) if args.resume else None

    if resume_path and (resume_path / "model").exists():
        # Check if it's a full fine-tuned checkpoint
        state_file = resume_path / "training_state.json"
        is_full_ft = False
        if state_file.exists():
            import json
            with open(state_file) as f:
                state = json.load(f)
            is_full_ft = state.get("full_finetune", False)

        if is_full_ft:
            # Load the full fine-tuned model as base
            model_path = str(resume_path / "model")
            print(f"\nLoading full fine-tuned SFT model from: {model_path}")
        else:
            print(f"\nLoading base model: {model_path} (will apply LoRA checkpoint after)")
    else:
        print(f"\nLoading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        spar_config.model.name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    # Load a frozen ref_model (same as the SFT checkpoint) for KL computation
    print("Loading reference model (frozen copy)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    # Create trainer (applies LoRA on top of model)
    print("\nCreating Offline GRPO trainer...")
    trainer = OfflineGRPOTrainer(
        config=offline_config,
        spar_config=spar_config,
        model=model,
        tokenizer=tokenizer,
        ref_model=ref_model,
        device=device,
    )

    # If resuming from a LoRA SFT checkpoint (not full FT), load LoRA weights
    is_full_ft_resume = resume_path is not None and (resume_path / "training_state.json").exists()
    if is_full_ft_resume:
        import json
        with open(resume_path / "training_state.json") as f:
            _state = json.load(f)
        is_full_ft_resume = _state.get("full_finetune", False)

    if resume_path and not is_full_ft_resume and (resume_path / "model").exists():
        print(f"Loading LoRA weights from {resume_path}")
        trainer.load_checkpoint(resume_path)

    # Train
    print("\n" + "=" * 60)
    print("Offline GRPO Training on Arena Debates")
    print("=" * 60)
    print(f"  Loss type:        {offline_config.loss_type}")
    print(f"  Epochs:           {offline_config.num_epochs}")
    print(f"  Batch size:       {offline_config.batch_size}")
    print(f"  Learning rate:    {offline_config.learning_rate}")
    print(f"  Topics:           {len(grouped)}")
    print(f"  Model filter:     {offline_config.model_filter}")
    print("=" * 60 + "\n")

    history = trainer.train(arena=arena, output_dir=args.output_dir)

    print("\nOffline GRPO training complete!")
    if history["loss"]:
        print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
