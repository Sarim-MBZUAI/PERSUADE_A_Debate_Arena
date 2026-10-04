#!/usr/bin/env python3
"""
Hybrid Training — Online GRPO with Dataset Opponents

Stage 3: Live generation + stored arena opponents + live judge evaluation.

Usage:
    python scripts/train_hybrid.py \
        --config configs/hybrid_arena.yaml \
        --model Qwen/Qwen3-4B \
        --opponent-model kimi-k2 \
        --resume outputs/offline_grpo/final \
        --output-dir outputs/hybrid
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spar.config import SPARConfig, load_config
from spar.data.arena_loader import ArenaDataLoader
from spar.training.hybrid_trainer import HybridTrainingConfig, HybridTrainer
from spar.debate.judge import create_judge
from spar.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid training with dataset opponents")

    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--model", type=str, default=None, help="Model override")
    parser.add_argument("--arena-data-dir", type=str, default="data/debate-arena")
    parser.add_argument("--opponent-model", type=str, default=None, help="Opponent model filter")
    parser.add_argument("--judge-type", type=str, choices=["openai", "local"], default=None)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/hybrid")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    judge_device = args.judge_device or (
        "cuda:1" if torch.cuda.device_count() > 1 else device
    )

    # Load config
    print(f"Loading config from {args.config}")
    spar_config = load_config(args.config)

    if args.model:
        spar_config.model.name_or_path = args.model

    # Judge overrides
    if args.judge_type:
        spar_config.judge.judge_type = args.judge_type
    if args.judge_model:
        if spar_config.judge.judge_type == "openai":
            spar_config.judge.model = args.judge_model
        else:
            spar_config.judge.local_model_path = args.judge_model

    # Hybrid config
    hybrid_config = HybridTrainingConfig()
    if args.opponent_model:
        hybrid_config.opponent_model_filter = args.opponent_model
    if args.epochs:
        hybrid_config.num_epochs = args.epochs
    if args.batch_size:
        hybrid_config.batch_size = args.batch_size
    if args.lr:
        hybrid_config.learning_rate = args.lr
    if args.num_generations:
        hybrid_config.num_generations = args.num_generations

    # Load arena data
    print(f"\nLoading arena data from {args.arena_data_dir}")
    arena = ArenaDataLoader(data_dir=args.arena_data_dir)
    stats = arena.stats()
    print(f"  Train debates: {stats['train_debates']}")
    print(f"  Train topics: {stats['train_topics']}")

    train_topics = arena.get_train_topics()

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

    # Create judge
    print(f"\nCreating judge ({spar_config.judge.judge_type})...")
    if spar_config.judge.judge_type == "openai":
        judge = create_judge(spar_config.judge)
    else:
        spar_config.judge.local_device = judge_device
        judge = create_judge(spar_config.judge, device=judge_device)

    # Create trainer
    print("\nCreating Hybrid trainer...")
    trainer = HybridTrainer(
        config=hybrid_config,
        spar_config=spar_config,
        model=model,
        tokenizer=tokenizer,
        judge=judge,
        arena=arena,
        device=device,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    # Train
    print("\n" + "=" * 60)
    print("Hybrid Training: Online GRPO + Dataset Opponents")
    print("=" * 60)
    print(f"  Opponent filter:  {hybrid_config.opponent_model_filter}")
    print(f"  Epochs:           {hybrid_config.num_epochs}")
    print(f"  Batch size:       {hybrid_config.batch_size}")
    print(f"  Learning rate:    {hybrid_config.learning_rate}")
    print(f"  Generations/K:    {hybrid_config.num_generations}")
    print(f"  Train topics:     {len(train_topics)}")
    print("=" * 60 + "\n")

    eval_topics = arena.get_test_topics()

    history = trainer.train(
        train_topics=train_topics,
        eval_topics=eval_topics,
        output_dir=args.output_dir,
    )

    print("\nHybrid training complete!")
    if history["win_rate"]:
        print(f"  Final win rate: {history['win_rate'][-1]:.2%}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
