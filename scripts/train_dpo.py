#!/usr/bin/env python3
"""
DPO Training on Arena Preference Pairs

Stage 4: Direct Preference Optimization using winning vs losing arguments.

Usage:
    python scripts/train_dpo.py \
        --config configs/dpo_arena.yaml \
        --model Qwen/Qwen3-4B \
        --chosen-model kimi-k2 \
        --mode dpo \
        --resume outputs/hybrid/final \
        --output-dir outputs/dpo
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spar.config import SPARConfig, load_config
from spar.data.arena_loader import ArenaDataLoader
from spar.training.dpo_trainer import DPOConfig, DPOTrainer
from spar.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training on arena preference pairs")

    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--model", type=str, default=None, help="Model override")
    parser.add_argument("--arena-data-dir", type=str, default="data/debate-arena")
    parser.add_argument("--chosen-model", type=str, default=None, help="Preferred model filter")
    parser.add_argument(
        "--mode", type=str, choices=["dpo", "filtered_grpo"], default=None,
    )
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/dpo")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
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

    # DPO config
    dpo_config = DPOConfig()
    if args.chosen_model:
        dpo_config.chosen_model_filter = args.chosen_model
    if args.mode:
        dpo_config.mode = args.mode
    if args.beta:
        dpo_config.beta = args.beta
    if args.epochs:
        dpo_config.num_epochs = args.epochs
    if args.batch_size:
        dpo_config.batch_size = args.batch_size
    if args.lr:
        dpo_config.learning_rate = args.lr

    # Load arena data
    print(f"\nLoading arena data from {args.arena_data_dir}")
    arena = ArenaDataLoader(data_dir=args.arena_data_dir)
    stats = arena.stats()
    print(f"  Train debates: {stats['train_debates']}")

    # Check preference pairs
    pairs = arena.get_preference_pairs(dpo_config.chosen_model_filter)
    print(f"  Preference pairs: {len(pairs)}")

    # Load model
    print(f"\nLoading model: {spar_config.model.name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        spar_config.model.name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spar_config.model.name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    # Create trainer
    print("\nCreating DPO trainer...")
    trainer = DPOTrainer(
        config=dpo_config,
        spar_config=spar_config,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    # Train
    print("\n" + "=" * 60)
    print("DPO Training on Arena Preference Pairs")
    print("=" * 60)
    print(f"  Mode:             {dpo_config.mode}")
    print(f"  Chosen model:     {dpo_config.chosen_model_filter}")
    print(f"  Beta:             {dpo_config.beta}")
    print(f"  Epochs:           {dpo_config.num_epochs}")
    print(f"  Batch size:       {dpo_config.batch_size}")
    print(f"  Learning rate:    {dpo_config.learning_rate}")
    print(f"  Preference pairs: {len(pairs)}")
    print("=" * 60 + "\n")

    history = trainer.train(arena=arena, output_dir=args.output_dir)

    print("\nDPO training complete!")
    if history["loss"]:
        print(f"  Final loss: {history['loss'][-1]:.4f}")
    if history.get("accuracy"):
        print(f"  Final accuracy: {history['accuracy'][-1]:.2%}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
