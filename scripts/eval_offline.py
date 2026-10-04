#!/usr/bin/env python3
"""
Offline Test-Set Evaluation

Computes perplexity and win-prediction accuracy on held-out test debates.
No text generation needed — just forward passes for log-prob computation.

Metrics:
  - Perplexity on winning arguments
  - Win-prediction accuracy (higher log-prob → predicted winner)
  - Per-model breakdown (which model's style was learned best)
  - Per-turn breakdown (performance across turn numbers)

Usage:
  python scripts/eval_offline.py \
      --checkpoint outputs/phase1_sft_full/best/model \
      --arena-data-dir data/debate-arena \
      --device cuda:0 \
      --output eval_results/sft_full_offline.json

  # LoRA checkpoint:
  python scripts/eval_offline.py \
      --checkpoint outputs/phase2_offline_grpo/final/model \
      --base-model outputs/phase1_sft_full/best/model \
      --arena-data-dir data/debate-arena \
      --device cuda:0 \
      --output eval_results/grpo.json
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spar.data.arena_loader import ArenaDataLoader, ArenaDebate, _strip_think_tags
from spar.utils.text import get_system_prompt, build_user_prompt, pseudo_plan_from_argument


def parse_args():
    parser = argparse.ArgumentParser(description="Offline test-set evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (full FT or LoRA adapter)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model for LoRA checkpoints. If not set, checkpoint is treated as full model.",
    )
    parser.add_argument(
        "--arena-data-dir",
        type=str,
        default="data/debate-arena",
        help="Path to debate-arena data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results/offline_eval.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--max-debates",
        type=int,
        default=None,
        help="Max debates to evaluate (for quick smoke tests)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for forward passes",
    )
    return parser.parse_args()


def load_model(checkpoint: str, base_model: str, device: str):
    """Load model, handling both full checkpoints and LoRA adapters."""
    model_path = Path(checkpoint)
    adapter_config = model_path / "adapter_config.json"

    # Determine tokenizer source
    tokenizer_path = base_model if base_model else checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_config.exists():
        # LoRA checkpoint
        from peft import PeftModel

        base_model_path = base_model if base_model else "Qwen/Qwen3-4B-Instruct-2507"
        print(f"Loading LoRA adapter from {checkpoint}")
        print(f"  Base model: {base_model_path}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, checkpoint)
    else:
        print(f"Loading full model from {checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def compute_log_probs(model, tokenizer, context: str, target: str) -> dict:
    """
    Compute log probabilities of target text given context.

    Returns dict with:
      - total_nll: negative log-likelihood summed over target tokens
      - num_tokens: number of target tokens
      - mean_nll: per-token NLL
      - perplexity: exp(mean_nll)
    """
    full_text = context + target
    full_inputs = tokenizer(
        full_text,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(model.device)

    context_inputs = tokenizer(context, return_tensors="pt")
    context_len = context_inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits

        # Shift: predict token[i+1] from logits[i]
        shift_logits = logits[:, :-1, :]
        shift_labels = full_inputs["input_ids"][:, 1:]

        log_probs_all = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs_all,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # Mask: only target tokens (after context)
        # context_len tokens produce context_len-1 shifted positions (indices 0..context_len-2)
        # Target starts at shifted index context_len-1
        target_start = max(context_len - 1, 0)
        target_log_probs = token_log_probs[0, target_start:]

        num_tokens = target_log_probs.numel()
        if num_tokens == 0:
            return {"total_nll": 0.0, "num_tokens": 0, "mean_nll": 0.0, "perplexity": 1.0}

        total_nll = -target_log_probs.sum().item()
        mean_nll = total_nll / num_tokens
        perplexity = math.exp(min(mean_nll, 100))  # cap to avoid overflow

    return {
        "total_nll": total_nll,
        "num_tokens": num_tokens,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
    }


def build_context_for_turn(
    debate: ArenaDebate,
    position: str,
    turn_num: int,
    tokenizer,
) -> str:
    """
    Build the context (system + user prompt with history) for a given turn.

    This reconstructs the prompt as the model would see it when generating
    the target turn's argument.
    """
    # Collect history up to (but not including) this turn's argument for this position
    history = []
    for t in debate.turns:
        # Include all turns before this one, plus same-turn entries
        # that come before this position speaks
        if t.turn < turn_num:
            history.append({
                "turn": t.turn,
                "agent": t.position,
                "argument": _strip_think_tags(t.argument),
            })
        elif t.turn == turn_num and t.position != position:
            # In a typical debate, FOR speaks first each turn.
            # If we're evaluating AGAINST at turn N, FOR's turn-N entry is in context.
            if position == "AGAINST" and t.position == "FOR":
                history.append({
                    "turn": t.turn,
                    "agent": t.position,
                    "argument": _strip_think_tags(t.argument),
                })

    system_prompt = get_system_prompt(position, include_plan=True)
    user_prompt = build_user_prompt(
        topic=debate.topic,
        position=position,
        turn=turn_num,
        debate_history=history,
        include_plan_instruction=True,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    context = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return context


def build_target_text(argument: str) -> str:
    """Build the target text (plan + debate tags) from an argument."""
    pseudo_plan = pseudo_plan_from_argument(argument)
    return f"<plan>\n{pseudo_plan}\n</plan>\n<debate>\n{argument}\n</debate>"


def evaluate_debate(
    model,
    tokenizer,
    debate: ArenaDebate,
) -> dict:
    """
    Evaluate a single debate. Computes log probs for both sides' arguments.

    Returns per-debate metrics including NLL for each side and win prediction.
    """
    winner_pos = debate.winner_position

    # Evaluate all turns
    for_total_nll = 0.0
    for_total_tokens = 0
    against_total_nll = 0.0
    against_total_tokens = 0

    per_turn_results = []

    for arena_turn in debate.turns:
        position = arena_turn.position
        turn_num = arena_turn.turn
        argument = _strip_think_tags(arena_turn.argument)

        context = build_context_for_turn(debate, position, turn_num, tokenizer)
        target = build_target_text(argument)

        result = compute_log_probs(model, tokenizer, context, target)
        result["position"] = position
        result["turn"] = turn_num
        result["model"] = arena_turn.model
        per_turn_results.append(result)

        if position == "FOR":
            for_total_nll += result["total_nll"]
            for_total_tokens += result["num_tokens"]
        else:
            against_total_nll += result["total_nll"]
            against_total_tokens += result["num_tokens"]

    # Win prediction: side with lower mean NLL (higher log-prob) is predicted winner
    for_mean_nll = for_total_nll / for_total_tokens if for_total_tokens > 0 else float("inf")
    against_mean_nll = against_total_nll / against_total_tokens if against_total_tokens > 0 else float("inf")

    predicted_winner = "FOR" if for_mean_nll <= against_mean_nll else "AGAINST"
    correct_prediction = predicted_winner == winner_pos

    # Winner's perplexity
    if winner_pos == "FOR":
        winner_nll = for_total_nll
        winner_tokens = for_total_tokens
    else:
        winner_nll = against_total_nll
        winner_tokens = against_total_tokens

    winner_mean_nll = winner_nll / winner_tokens if winner_tokens > 0 else 0.0
    winner_perplexity = math.exp(min(winner_mean_nll, 100))

    return {
        "topic": debate.topic,
        "winner_position": winner_pos,
        "winner_model": debate.winner_model,
        "for_model": debate.for_model,
        "against_model": debate.against_model,
        "for_mean_nll": for_mean_nll,
        "against_mean_nll": against_mean_nll,
        "winner_perplexity": winner_perplexity,
        "predicted_winner": predicted_winner,
        "correct_prediction": correct_prediction,
        "per_turn": per_turn_results,
    }


def main():
    args = parse_args()
    start_time = time.time()

    # Load model
    print("=" * 60)
    print("Offline Test-Set Evaluation")
    print("=" * 60)
    model, tokenizer = load_model(args.checkpoint, args.base_model, args.device)

    # Load arena test data
    print(f"\nLoading arena data from {args.arena_data_dir}...")
    loader = ArenaDataLoader(args.arena_data_dir, load_train=False, load_test=True)
    test_debates = loader.test_debates

    if args.max_debates is not None:
        test_debates = test_debates[:args.max_debates]

    print(f"  Test debates: {len(test_debates)}")
    print(f"  Test topics: {len(loader.test_topics)}")

    # Evaluate each debate
    all_results = []
    correct = 0
    total_winner_nll = 0.0
    total_winner_tokens = 0

    # Per-model tracking
    model_nlls = defaultdict(lambda: {"nll": 0.0, "tokens": 0, "count": 0})
    # Per-turn tracking
    turn_nlls = defaultdict(lambda: {"nll": 0.0, "tokens": 0, "count": 0})

    print(f"\nEvaluating {len(test_debates)} debates...\n")

    for i, debate in enumerate(test_debates):
        result = evaluate_debate(model, tokenizer, debate)
        all_results.append(result)

        if result["correct_prediction"]:
            correct += 1

        # Aggregate winner NLL
        for tr in result["per_turn"]:
            # Winner's turns
            if tr["position"] == result["winner_position"]:
                total_winner_nll += tr["total_nll"]
                total_winner_tokens += tr["num_tokens"]

            # Per-model
            model_name = tr["model"]
            model_nlls[model_name]["nll"] += tr["total_nll"]
            model_nlls[model_name]["tokens"] += tr["num_tokens"]
            model_nlls[model_name]["count"] += 1

            # Per-turn
            turn_key = tr["turn"]
            turn_nlls[turn_key]["nll"] += tr["total_nll"]
            turn_nlls[turn_key]["tokens"] += tr["num_tokens"]
            turn_nlls[turn_key]["count"] += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(test_debates):
            accuracy = correct / (i + 1) * 100
            avg_ppl = math.exp(min(total_winner_nll / max(total_winner_tokens, 1), 100))
            elapsed = time.time() - start_time
            print(
                f"  [{i+1}/{len(test_debates)}] "
                f"Accuracy: {accuracy:.1f}% | "
                f"Winner PPL: {avg_ppl:.2f} | "
                f"Time: {elapsed:.0f}s"
            )

    # Compute final aggregates
    total = len(all_results)
    accuracy = correct / total * 100 if total > 0 else 0.0
    overall_winner_ppl = math.exp(min(total_winner_nll / max(total_winner_tokens, 1), 100))

    # Per-model breakdown
    per_model = {}
    for model_name, stats in sorted(model_nlls.items()):
        mean_nll = stats["nll"] / stats["tokens"] if stats["tokens"] > 0 else 0.0
        per_model[model_name] = {
            "perplexity": math.exp(min(mean_nll, 100)),
            "mean_nll": mean_nll,
            "num_turns": stats["count"],
            "num_tokens": stats["tokens"],
        }

    # Per-turn breakdown
    per_turn = {}
    for turn_num in sorted(turn_nlls.keys()):
        stats = turn_nlls[turn_num]
        mean_nll = stats["nll"] / stats["tokens"] if stats["tokens"] > 0 else 0.0
        per_turn[turn_num] = {
            "perplexity": math.exp(min(mean_nll, 100)),
            "mean_nll": mean_nll,
            "num_entries": stats["count"],
            "num_tokens": stats["tokens"],
        }

    elapsed = time.time() - start_time

    # Summary
    summary = {
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "num_debates": total,
        "win_prediction_accuracy": accuracy,
        "winner_perplexity": overall_winner_ppl,
        "per_model": per_model,
        "per_turn": per_turn,
        "eval_time_seconds": elapsed,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Checkpoint:             {args.checkpoint}")
    print(f"  Test debates:           {total}")
    print(f"  Win-prediction accuracy: {accuracy:.1f}%")
    print(f"  Winner perplexity:      {overall_winner_ppl:.2f}")
    print(f"  Eval time:              {elapsed:.0f}s")

    print(f"\n  Per-model perplexity:")
    for model_name, stats in sorted(per_model.items(), key=lambda x: x[1]["perplexity"]):
        print(f"    {model_name:30s}  PPL={stats['perplexity']:8.2f}  (n={stats['num_turns']})")

    print(f"\n  Per-turn perplexity:")
    for turn_num, stats in sorted(per_turn.items()):
        print(f"    Turn {turn_num:2d}:  PPL={stats['perplexity']:8.2f}  (n={stats['num_entries']})")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "summary": summary,
        "debates": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
