"""
Arena Data Loader for SPAR

Loads debate-arena-data JSON files and converts them to SPAR-compatible
formats for training (offline demonstrations) and evaluation.

Dataset: 10-model LLM debate arena
- 10,487 train / 3,239 test debates
- 200 train / 59 test topics (no overlap)
- 10-turn debates judged by GPT-5.2
- Models: kimi-k2, deepseek-r1, qwen3-235b, glm-4p7, etc.
"""

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ..debate.action import DebateAction
from ..debate.environment import DebateResult
from ..utils.text import extract_plan, extract_utterance, pseudo_plan_from_argument


@dataclass
class ArenaTurn:
    """Single turn entry from arena debate."""
    turn: int
    position: str  # "FOR" or "AGAINST"
    model: str
    argument: str


@dataclass
class ArenaDebate:
    """Parsed arena debate with metadata."""
    topic: str
    for_model: str
    against_model: str
    winner: str  # "A" or "B" label
    winner_model: str
    loser_model: str
    turns: List[ArenaTurn]
    filename: str = ""

    @property
    def num_turns(self) -> int:
        """Number of debate rounds (each round = 1 FOR + 1 AGAINST)."""
        return max(t.turn for t in self.turns) if self.turns else 0

    @property
    def winner_position(self) -> str:
        """Position of the winner ('FOR' or 'AGAINST')."""
        if self.winner_model == self.for_model:
            return "FOR"
        return "AGAINST"

    @property
    def for_turns(self) -> List[ArenaTurn]:
        return [t for t in self.turns if t.position == "FOR"]

    @property
    def against_turns(self) -> List[ArenaTurn]:
        return [t for t in self.turns if t.position == "AGAINST"]

    def to_debate_result(self, max_turns: Optional[int] = None) -> DebateResult:
        """
        Convert to SPAR DebateResult format.

        Args:
            max_turns: If set, truncate debate to this many rounds.
                       Useful for curriculum learning (1->2->3->4 turns).
        """
        for_turns = self.for_turns
        against_turns = self.against_turns

        if max_turns is not None:
            for_turns = for_turns[:max_turns]
            against_turns = against_turns[:max_turns]

        # Build history in alternating order
        history = []
        for i in range(max(len(for_turns), len(against_turns))):
            if i < len(for_turns):
                ft = for_turns[i]
                history.append({
                    "turn": ft.turn,
                    "agent": "FOR",
                    "argument": _strip_think_tags(ft.argument),
                })
            if i < len(against_turns):
                at = against_turns[i]
                history.append({
                    "turn": at.turn,
                    "agent": "AGAINST",
                    "argument": _strip_think_tags(at.argument),
                })

        # Build DebateActions (arena data has no explicit plans, so we
        # create pseudo-plans from the first 2 sentences of each argument)
        for_actions = []
        for ft in for_turns:
            arg = _strip_think_tags(ft.argument)
            for_actions.append(DebateAction(
                plan=pseudo_plan_from_argument(arg),
                utterance=arg,
                position="FOR",
                turn=ft.turn,
                raw_text=ft.argument,
            ))

        against_actions = []
        for at in against_turns:
            arg = _strip_think_tags(at.argument)
            against_actions.append(DebateAction(
                plan=pseudo_plan_from_argument(arg),
                utterance=arg,
                position="AGAINST",
                turn=at.turn,
                raw_text=at.argument,
            ))

        # Determine winner position
        winner_pos = self.winner_position

        return DebateResult(
            topic=self.topic,
            domain="",
            judge_initial_position="NEUTRAL",
            judge_initial_belief=0.5,
            judge_final_decision=winner_pos,
            winner=winner_pos,
            history=history,
            for_actions=for_actions,
            against_actions=against_actions,
            intermediate_beliefs=[],
            metadata={
                "source": "debate-arena",
                "for_model": self.for_model,
                "against_model": self.against_model,
                "winner_model": self.winner_model,
                "loser_model": self.loser_model,
                "filename": self.filename,
            },
        )


class ArenaDataLoader:
    """
    Loads and manages debate-arena benchmark data for SPAR training.

    Supports:
    - Loading train/test splits
    - Filtering by topic, model, winner
    - Converting to SPAR DebateResult format
    - Sampling for online training (topic pool)
    - Extracting demonstration trajectories for offline RL
    """

    def __init__(
        self,
        data_dir: str = "data/debate-arena",
        load_train: bool = True,
        load_test: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.train_debates: List[ArenaDebate] = []
        self.test_debates: List[ArenaDebate] = []
        self.train_topics: List[str] = []
        self.test_topics: List[str] = []
        self.train_summary: Dict[str, Any] = {}
        self.test_summary: Dict[str, Any] = {}

        if load_train:
            self._load_split("train")
        if load_test:
            self._load_split("test")

    def _load_split(self, split: str) -> None:
        """Load all debates for a split."""
        split_dir = self.data_dir / split
        debates_dir = split_dir / "debates"
        summary_path = split_dir / "summary.json"

        # Load summary
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            if split == "train":
                self.train_summary = summary
                self.train_topics = summary.get("topics", [])
            else:
                self.test_summary = summary
                self.test_topics = summary.get("topics", [])

        # Load individual debates
        if not debates_dir.exists():
            return

        debates = []
        for fname in sorted(os.listdir(debates_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = debates_dir / fname
            try:
                with open(fpath) as f:
                    raw = json.load(f)
                debate = self._parse_debate(raw, fname)
                debates.append(debate)
            except (json.JSONDecodeError, KeyError) as e:
                continue

        if split == "train":
            self.train_debates = debates
        else:
            self.test_debates = debates

    def _parse_debate(self, raw: Dict[str, Any], filename: str) -> ArenaDebate:
        """Parse raw JSON into ArenaDebate."""
        turns = []
        for entry in raw.get("debate", []):
            turns.append(ArenaTurn(
                turn=entry["turn"],
                position=entry["position"],
                model=entry["model"],
                argument=entry["argument"],
            ))

        return ArenaDebate(
            topic=raw["topic"],
            for_model=raw["for_model"],
            against_model=raw["against_model"],
            winner=raw["winner"],
            winner_model=raw["winner_model"],
            loser_model=raw["loser_model"],
            turns=turns,
            filename=filename,
        )

    # --- Topic pool for online training ---

    def get_train_topics(self) -> List[str]:
        """Get all training topics (for online SPAR training)."""
        return list(self.train_topics)

    def get_test_topics(self) -> List[str]:
        """Get all test topics (for evaluation)."""
        return list(self.test_topics)

    def sample_topics(
        self,
        n: int,
        split: str = "train",
        seed: Optional[int] = None,
    ) -> List[str]:
        """Sample n topics from the given split."""
        topics = self.train_topics if split == "train" else self.test_topics
        rng = random.Random(seed)
        return rng.sample(topics, min(n, len(topics)))

    # --- Demonstration data for offline RL ---

    def get_winning_demonstrations(
        self,
        position: str = "FOR",
        model_filter: Optional[str] = None,
        max_turns: Optional[int] = None,
        split: str = "train",
    ) -> List[DebateResult]:
        """
        Get debates where the specified position won.

        Useful as expert demonstrations for:
        - Behavioral cloning warmup
        - Offline RL reference trajectories
        - Reward model training data

        Args:
            position: "FOR" or "AGAINST" — get debates where this side won
            model_filter: If set, only include debates involving this model
            max_turns: Truncate debates to this many rounds
            split: "train" or "test"
        """
        debates = self.train_debates if split == "train" else self.test_debates
        results = []

        for debate in debates:
            # Filter by winner position
            if debate.winner_position != position:
                continue

            # Filter by model
            if model_filter:
                if model_filter not in debate.for_model and model_filter not in debate.against_model:
                    continue

            results.append(debate.to_debate_result(max_turns=max_turns))

        return results

    def get_debates_for_topic(
        self,
        topic: str,
        split: str = "train",
        max_debates: Optional[int] = None,
    ) -> List[DebateResult]:
        """Get all debates for a specific topic."""
        debates = self.train_debates if split == "train" else self.test_debates
        results = []

        for debate in debates:
            if debate.topic == topic:
                results.append(debate.to_debate_result())
                if max_debates and len(results) >= max_debates:
                    break

        return results

    def get_debates_by_model(
        self,
        model: str,
        position: Optional[str] = None,
        split: str = "train",
    ) -> List[DebateResult]:
        """Get debates involving a specific model."""
        debates = self.train_debates if split == "train" else self.test_debates
        results = []

        for debate in debates:
            if position == "FOR" and model not in debate.for_model:
                continue
            if position == "AGAINST" and model not in debate.against_model:
                continue
            if position is None and model not in debate.for_model and model not in debate.against_model:
                continue
            results.append(debate.to_debate_result())

        return results

    def get_model_rankings(self, split: str = "train") -> List[Tuple[str, int]]:
        """Get model rankings by win count."""
        summary = self.train_summary if split == "train" else self.test_summary
        return summary.get("rankings", [])

    def get_win_rates(self, split: str = "train") -> Dict[str, Dict[str, Optional[float]]]:
        """Get head-to-head win rates."""
        summary = self.train_summary if split == "train" else self.test_summary
        return summary.get("win_rates", {})

    # --- Batch iteration for training ---

    def iter_train_batches(
        self,
        batch_size: int = 8,
        max_turns: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Iterate over training debates in batches.

        Yields batches of DebateResult for offline training.
        """
        debates = list(self.train_debates)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(debates)

        batch = []
        for debate in debates:
            batch.append(debate.to_debate_result(max_turns=max_turns))
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    # --- Expert demonstrations for SFT ---

    def get_expert_demonstrations(
        self,
        expert_models: List[str],
        include_beaters: bool = True,
        split: str = "train",
    ) -> List[ArenaDebate]:
        """
        Get expert-level debate demonstrations for SFT training.

        Sources:
          (a) All debates won by expert_models
          (b) If include_beaters: debates where any model beat an expert
              (the winner's turns are also expert-quality)

        Args:
            expert_models: Model names to treat as experts (e.g. ["kimi-k2"])
            include_beaters: Also include debates where expert lost (winner is strong)
            split: "train" or "test"

        Returns:
            List of ArenaDebate objects matching the criteria
        """
        debates = self.train_debates if split == "train" else self.test_debates
        results = []
        seen = set()

        for debate in debates:
            # (a) Expert won this debate
            if any(expert in debate.winner_model for expert in expert_models):
                if debate.filename not in seen:
                    results.append(debate)
                    seen.add(debate.filename)
                continue

            # (b) Expert lost — the winner beat the expert, so winner is strong
            if include_beaters:
                is_expert_debate = any(
                    expert in debate.for_model or expert in debate.against_model
                    for expert in expert_models
                )
                if is_expert_debate and debate.filename not in seen:
                    results.append(debate)
                    seen.add(debate.filename)

        return results

    # --- Grouped topics for offline GRPO ---

    def get_debates_grouped_by_topic(
        self,
        split: str = "train",
        min_per_topic: int = 4,
        model_filter: Optional[str] = None,
    ) -> Dict[str, List[ArenaDebate]]:
        """
        Group debates by topic with minimum count filtering.

        Args:
            split: "train" or "test"
            min_per_topic: Minimum debates per topic to include
            model_filter: If set, only include debates involving this model

        Returns:
            Dict mapping topic -> list of ArenaDebate
        """
        debates = self.train_debates if split == "train" else self.test_debates
        groups: Dict[str, List[ArenaDebate]] = {}

        for debate in debates:
            if model_filter:
                if model_filter not in debate.for_model and model_filter not in debate.against_model:
                    continue
            groups.setdefault(debate.topic, []).append(debate)

        # Filter by minimum count
        return {t: ds for t, ds in groups.items() if len(ds) >= min_per_topic}

    # --- Preference pairs for DPO ---

    def get_preference_pairs(
        self,
        chosen_model: str,
        split: str = "train",
    ) -> List[Dict]:
        """
        Build preference pairs for DPO training.

        For each topic+position+turn, pairs a chosen argument (from the winning
        side of a debate involving chosen_model) with a rejected argument (from
        the losing side on the same topic+position+turn).

        Args:
            chosen_model: Model whose winning arguments are preferred
            split: "train" or "test"

        Returns:
            List of dicts with keys: topic, position, turn, chosen_argument,
            rejected_argument, chosen_model, rejected_model
        """
        debates = self.train_debates if split == "train" else self.test_debates

        # Index arguments by (topic, position, turn)
        winning_args: Dict[tuple, List[Dict]] = {}
        losing_args: Dict[tuple, List[Dict]] = {}

        for debate in debates:
            involves_chosen = (
                chosen_model in debate.for_model or chosen_model in debate.against_model
            )
            if not involves_chosen:
                continue

            for t in debate.turns:
                key = (debate.topic, t.position, t.turn)
                entry = {
                    "argument": _strip_think_tags(t.argument),
                    "model": t.model,
                    "winner_model": debate.winner_model,
                }

                is_winner_side = (
                    (t.position == "FOR" and debate.winner_model == debate.for_model) or
                    (t.position == "AGAINST" and debate.winner_model == debate.against_model)
                )

                if is_winner_side:
                    winning_args.setdefault(key, []).append(entry)
                else:
                    losing_args.setdefault(key, []).append(entry)

        # Build pairs
        pairs = []
        for key in winning_args:
            if key not in losing_args:
                continue
            for win_entry in winning_args[key]:
                for lose_entry in losing_args[key]:
                    if win_entry["argument"] == lose_entry["argument"]:
                        continue
                    pairs.append({
                        "topic": key[0],
                        "position": key[1],
                        "turn": key[2],
                        "chosen_argument": win_entry["argument"],
                        "rejected_argument": lose_entry["argument"],
                        "chosen_model": win_entry["model"],
                        "rejected_model": lose_entry["model"],
                    })

        return pairs

    # --- Opponent responses for hybrid training ---

    def get_opponent_responses(
        self,
        topic: str,
        position: str,
        model_filter: Optional[str] = None,
        split: str = "train",
    ) -> Dict[int, List[str]]:
        """
        Get stored opponent responses for a topic, indexed by turn.

        Args:
            topic: Debate topic
            position: Position of the opponent ("FOR" or "AGAINST")
            model_filter: Only include responses from this model
            split: "train" or "test"

        Returns:
            Dict mapping turn_number -> list of argument strings
        """
        debates = self.train_debates if split == "train" else self.test_debates
        responses: Dict[int, List[str]] = {}

        for debate in debates:
            if debate.topic != topic:
                continue

            for t in debate.turns:
                if t.position != position:
                    continue
                if model_filter and model_filter not in t.model:
                    continue
                responses.setdefault(t.turn, []).append(
                    _strip_think_tags(t.argument)
                )

        return responses

    # --- Stats ---

    def stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "train_debates": len(self.train_debates),
            "test_debates": len(self.test_debates),
            "train_topics": len(self.train_topics),
            "test_topics": len(self.test_topics),
            "models": self.train_summary.get("models", []),
            "judge": self.train_summary.get("judge", "unknown"),
            "train_rankings": self.get_model_rankings("train"),
        }


def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning tags from model outputs."""
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip() or text.strip()
