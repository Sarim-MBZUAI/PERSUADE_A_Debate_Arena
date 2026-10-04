"""
Debate Action Representation

Hierarchical action: Plan → Utterance
"""

from dataclasses import dataclass
from typing import Optional
import torch
import re


@dataclass
class DebateAction:
    """
    Hierarchical action in the debate.

    Each action consists of:
    1. plan: High-level strategy (internal, not shown to opponent)
    2. utterance: Actual debate argument (shown to opponent and judge)

    The plan is marked with <plan>...</plan> tags.
    The utterance is marked with <debate>...</debate> tags.
    """

    plan: str
    utterance: str
    position: str  # "FOR" or "AGAINST"
    turn: int

    # Token-level log probabilities for RL training
    plan_log_probs: Optional[torch.Tensor] = None
    utterance_log_probs: Optional[torch.Tensor] = None
    full_log_probs: Optional[torch.Tensor] = None

    # Token indices for masking
    plan_token_indices: Optional[tuple] = None  # (start, end)
    utterance_token_indices: Optional[tuple] = None  # (start, end)

    # Raw generated text (before parsing)
    raw_text: Optional[str] = None

    @property
    def total_log_prob(self) -> Optional[torch.Tensor]:
        """Sum of plan and utterance log probs."""
        if self.full_log_probs is not None:
            return self.full_log_probs.sum()
        if self.plan_log_probs is not None and self.utterance_log_probs is not None:
            return self.plan_log_probs.sum() + self.utterance_log_probs.sum()
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "turn": self.turn,
            "agent": self.position,
            "plan": self.plan,
            "argument": self.utterance,
        }

    @classmethod
    def from_raw_text(
        cls,
        text: str,
        position: str,
        turn: int,
        log_probs: Optional[torch.Tensor] = None,
    ) -> "DebateAction":
        """
        Parse action from raw model output.

        Expected format:
        <plan>
        - bullet point 1
        - bullet point 2
        </plan>
        <debate>
        Argument text here.
        </debate>
        """
        plan = cls._extract_plan(text)
        utterance = cls._extract_utterance(text)

        return cls(
            plan=plan,
            utterance=utterance,
            position=position,
            turn=turn,
            full_log_probs=log_probs,
            raw_text=text,
        )

    @staticmethod
    def _extract_plan(text: str) -> str:
        """Extract plan from <plan>...</plan> tags."""
        pattern = r"<plan>(.*?)</plan>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: look for bullet points at the start
        lines = text.strip().split("\n")
        plan_lines = []
        for line in lines:
            if line.strip().startswith(("-", "*", "•")):
                plan_lines.append(line)
            elif plan_lines:  # Stop when we hit non-bullet line after bullets
                break

        if plan_lines:
            return "\n".join(plan_lines)

        return ""

    @staticmethod
    def _extract_utterance(text: str) -> str:
        """Extract utterance from <debate>...</debate> tags."""
        pattern = r"<debate>(.*?)</debate>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: remove plan tags and take remaining text
        cleaned = re.sub(r"<plan>.*?</plan>", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"</?debate>", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        # If still nothing, just take the text after plan section
        if not cleaned:
            lines = text.strip().split("\n")
            # Skip bullet points
            non_bullet_lines = []
            in_plan = True
            for line in lines:
                if line.strip().startswith(("-", "*", "•")):
                    continue
                elif line.strip():
                    in_plan = False
                    non_bullet_lines.append(line)
                elif not in_plan:
                    non_bullet_lines.append(line)
            cleaned = "\n".join(non_bullet_lines).strip()

        return cleaned if cleaned else text.strip()

    @classmethod
    def create_with_separate_generation(
        cls,
        plan: str,
        utterance: str,
        position: str,
        turn: int,
        plan_log_probs: Optional[torch.Tensor] = None,
        utterance_log_probs: Optional[torch.Tensor] = None,
    ) -> "DebateAction":
        """Create action when plan and utterance are generated separately."""
        # Construct raw_text to match format from from_raw_text
        raw_text = f"<plan>\n{plan}\n</plan>\n<debate>\n{utterance}\n</debate>"
        return cls(
            plan=plan,
            utterance=utterance,
            position=position,
            turn=turn,
            plan_log_probs=plan_log_probs,
            utterance_log_probs=utterance_log_probs,
            raw_text=raw_text,
        )

    def format_for_history(self) -> str:
        """Format action for inclusion in debate history (without plan)."""
        return self.utterance

    def format_with_plan(self) -> str:
        """Format action with plan visible (for training)."""
        return f"<plan>\n{self.plan}\n</plan>\n<debate>\n{self.utterance}\n</debate>"

    def __repr__(self) -> str:
        plan_preview = self.plan[:50] + "..." if len(self.plan) > 50 else self.plan
        utt_preview = self.utterance[:50] + "..." if len(self.utterance) > 50 else self.utterance
        return (
            f"DebateAction(position={self.position}, turn={self.turn}, "
            f"plan='{plan_preview}', utterance='{utt_preview}')"
        )


def format_plan_prompt(
    state_text: str,
    include_examples: bool = False,
) -> str:
    """
    Format prompt for plan generation.

    Args:
        state_text: Current state as text
        include_examples: Whether to include few-shot examples

    Returns:
        Formatted prompt for plan generation
    """
    prompt = f"""{state_text}

Create a strategic plan for your next argument. The plan should include:
- 3-6 key points or strategies you will use
- 1-2 anticipated counterarguments to address

Format your plan as bullet points inside <plan>...</plan> tags.
Then write your argument inside <debate>...</debate> tags.

<plan>"""

    return prompt


def format_utterance_prompt(
    state_text: str,
    plan: str,
) -> str:
    """
    Format prompt for utterance generation given a plan.

    Args:
        state_text: Current state as text
        plan: The selected plan

    Returns:
        Formatted prompt for utterance generation
    """
    prompt = f"""{state_text}

Your strategic plan:
<plan>
{plan}
</plan>

Now write your persuasive argument based on this plan.
Keep it concise but impactful (3-5 sentences).
Address your opponent's points when relevant.

<debate>"""

    return prompt
