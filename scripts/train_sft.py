#!/usr/bin/env python3
"""
SFT Training on Expert Debate Demonstrations

Stage 1: Supervised fine-tuning on winning arguments from expert models.

Usage:
    python scripts/train_sft.py \
        --config configs/sft_arena.yaml \
        --model Qwen/Qwen3-4B \
        --expert-models kimi-k2 \
        --output-dir outputs/sft
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spar.config import SPARConfig, load_config
from spar.data.arena_loader import ArenaDataLoader
from spar.training.sft_trainer import SFTConfig, SFTTrainer, DebateSFTDataset
from spar.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training on expert debates")

    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--model", type=str, default=None, help="Model override")
    parser.add_argument("--arena-data-dir", type=str, default="data/debate-arena")
    parser.add_argument(
        "--expert-models", type=str, nargs="+", default=None,
        help="Expert model names (e.g. kimi-k2)",
    )
    parser.add_argument("--include-kimi-beaters", action="store_true", default=True)
    parser.add_argument("--no-kimi-beaters", dest="include_kimi_beaters", action="store_false")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/sft")
    parser.add_argument("--full-finetune", action="store_true", default=False,
                        help="Full fine-tuning instead of LoRA")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load SPAR config
    print(f"Loading config from {args.config}")
    spar_config = load_config(args.config)

    if args.model:
        spar_config.model.name_or_path = args.model

    # Build SFT config
    sft_config = SFTConfig()
    if args.expert_models:
        sft_config.expert_models = args.expert_models
    sft_config.include_kimi_beaters = args.include_kimi_beaters
    if args.epochs:
        sft_config.num_epochs = args.epochs
    if args.batch_size:
        sft_config.batch_size = args.batch_size
    if args.lr:
        sft_config.learning_rate = args.lr
    if args.max_seq_length:
        sft_config.max_seq_length = args.max_seq_length
    sft_config.full_finetune = args.full_finetune

    # Load arena data
    print(f"\nLoading arena data from {args.arena_data_dir}")
    arena = ArenaDataLoader(data_dir=args.arena_data_dir)
    stats = arena.stats()
    print(f"  Train debates: {stats['train_debates']}")
    print(f"  Train topics: {stats['train_topics']}")

    # Load model & tokenizer
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

    # Build dataset
    print(f"\nBuilding SFT dataset (max_seq_length={sft_config.max_seq_length})...")
    train_dataset = DebateSFTDataset.from_arena_data(
        arena, tokenizer, sft_config, split="train",
    )
    print(f"  Raw training examples: {len(train_dataset)}")
    removed = train_dataset.filter_valid_examples()
    print(f"  After filtering (removed {removed} too-long prompts): {len(train_dataset)}")

    eval_dataset = None
    if arena.test_debates:
        eval_dataset = DebateSFTDataset.from_arena_data(
            arena, tokenizer, sft_config, split="test",
        )
        eval_removed = eval_dataset.filter_valid_examples()
        print(f"  Eval examples: {len(eval_dataset)} (removed {eval_removed})")

    # Create trainer
    print("\nCreating SFT trainer...")
    trainer = SFTTrainer(
        config=sft_config,
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
    print("SFT Training on Expert Demonstrations")
    print("=" * 60)
    print(f"  Mode:             {'Full Fine-Tuning' if sft_config.full_finetune else 'LoRA (r={})'.format(spar_config.model.lora_r)}")
    print(f"  Expert models:    {sft_config.expert_models}")
    print(f"  Include beaters:  {sft_config.include_kimi_beaters}")
    print(f"  Epochs:           {sft_config.num_epochs}")
    print(f"  Batch size:       {sft_config.batch_size}")
    print(f"  Learning rate:    {sft_config.learning_rate}")
    print(f"  Training examples: {len(train_dataset)}")
    print("=" * 60 + "\n")

    history = trainer.train(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("SFT Training complete!")
    if history["loss"]:
        print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
