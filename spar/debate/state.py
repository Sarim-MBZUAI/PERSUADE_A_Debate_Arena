"""
Debate State Representation

Immutable state for tracking debate progress.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import hashlib


@dataclass(frozen=True)
class DebateState:
    """
    Immutable state representation of a debate at any point.

    The state includes the topic, current position, turn number,
    and full debate history as an immutable tuple.
    """

    topic: str
    domain: str
    position: str  # "FOR" or "AGAINST"
    turn: int
    history: Tuple[Dict[str, Any], ...]  # Immutable history
    judge_initial_position: Optional[str] = None
    judge_initial_belief: Optional[float] = None  # 0-1 probability

    def to_prompt(
        self,
        include_plan_instruction: bool = True,
        max_history_turns: Optional[int] = None,
    ) -> str:
        """
        Format state for model input.

        Args:
            include_plan_instruction: Whether to include plan generation instruction
            max_history_turns: Maximum number of history turns to include

        Returns:
            Formatted prompt string
        """
        parts = [f"Topic: {self.topic}"]

        if self.domain:
            parts.append(f"Domain: {self.domain}")

        parts.append(f"Your position: {self.position}")
        parts.append(f"Current turn: {self.turn}")

        # Add debate history
        if self.history:
            history_str = self._format_history(max_history_turns)
            if history_str:
                parts.append(f"\nDEBATE SO FAR:\n{history_str}")

        prompt = "\n".join(parts)

        if include_plan_instruction:
            prompt += "\n\nFirst create your strategic plan, then your argument."

        return prompt

    def _format_history(self, max_turns: Optional[int] = None) -> str:
        """Format debate history as string."""
        history = self.history
        if max_turns is not None and len(history) > max_turns * 2:
            # Take last max_turns exchanges
            history = history[-(max_turns * 2):]

        lines = []
        for entry in history:
            agent = entry.get("agent", "UNKNOWN")
            argument = entry.get("argument", "")
            turn = entry.get("turn", "?")
            lines.append(f"Turn {turn} - {agent}: {argument}")

        return "\n".join(lines)

    def to_state_text(self, max_history_turns: Optional[int] = 6) -> str:
        """
        Create compact state text for critic/reward computation.

        Similar to pnlc/text.py build_state_text.
        """
        parts = [
            f"TOPIC: {self.topic}",
            f"POSITION: {self.position}",
        ]

        if self.history:
            parts.append("DEBATE SO FAR:")
            history_str = self._format_history(max_history_turns)
            parts.append(history_str)

        return "\n".join(parts)

    def append_entry(
        self,
        agent: str,
        argument: str,
        plan: Optional[str] = None,
        turn: Optional[int] = None,
    ) -> "DebateState":
        """
        Return new state with appended history entry.

        Args:
            agent: "FOR" or "AGAINST"
            argument: The debate argument text
            plan: Optional plan text
            turn: Turn number (defaults to current turn)

        Returns:
            New DebateState with updated history
        """
        entry = {
            "turn": turn if turn is not None else self.turn,
            "agent": agent,
            "argument": argument,
        }
        if plan is not None:
            entry["plan"] = plan

        new_history = self.history + (entry,)

        return DebateState(
            topic=self.topic,
            domain=self.domain,
            position=self.position,
            turn=self.turn,
            history=new_history,
            judge_initial_position=self.judge_initial_position,
            judge_initial_belief=self.judge_initial_belief,
        )

    def with_position(self, position: str) -> "DebateState":
        """Return new state with different position (for opponent)."""
        return DebateState(
            topic=self.topic,
            domain=self.domain,
            position=position,
            turn=self.turn,
            history=self.history,
            judge_initial_position=self.judge_initial_position,
            judge_initial_belief=self.judge_initial_belief,
        )

    def with_turn(self, turn: int) -> "DebateState":
        """Return new state with different turn number."""
        return DebateState(
            topic=self.topic,
            domain=self.domain,
            position=self.position,
            turn=turn,
            history=self.history,
            judge_initial_position=self.judge_initial_position,
            judge_initial_belief=self.judge_initial_belief,
        )

    def with_judge_position(
        self,
        position: str,
        belief: Optional[float] = None,
    ) -> "DebateState":
        """Return new state with judge's initial position."""
        return DebateState(
            topic=self.topic,
            domain=self.domain,
            position=self.position,
            turn=self.turn,
            history=self.history,
            judge_initial_position=position,
            judge_initial_belief=belief,
        )

    @property
    def num_entries(self) -> int:
        """Number of entries in history."""
        return len(self.history)

    @property
    def num_complete_turns(self) -> int:
        """Number of complete turns (both FOR and AGAINST have spoken)."""
        return self.num_entries // 2

    @property
    def state_hash(self) -> str:
        """Get hash of state for caching."""
        content = f"{self.topic}|{self.position}|{self.turn}|{len(self.history)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    @classmethod
    def from_topic(
        cls,
        topic: str,
        domain: str = "",
        position: str = "FOR",
    ) -> "DebateState":
        """Create initial state from topic."""
        return cls(
            topic=topic,
            domain=domain,
            position=position,
            turn=1,
            history=(),
            judge_initial_position=None,
            judge_initial_belief=None,
        )

    def __repr__(self) -> str:
        return (
            f"DebateState(topic={self.topic[:30]}..., "
            f"position={self.position}, "
            f"turn={self.turn}, "
            f"history_len={len(self.history)})"
        )
