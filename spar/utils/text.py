"""
Text Processing Utilities

Prompt templates, state formatting, and text manipulation.
"""

import re
from typing import List, Dict, Any, Optional


# System prompts for different agent configurations
SYSTEM_PROMPTS = {
    "for_with_plan": """You are a skilled debater arguing FOR the following topic.
Provide well-reasoned, persuasive arguments. Address your opponent's points when relevant.

First, create a strategic plan inside <plan>...</plan> tags:
- 3-6 key points or strategies for your argument
- 1-2 anticipated counterarguments to address

Then, write your persuasive argument inside <debate>...</debate> tags.
Keep your response concise but impactful (3-5 sentences max).""",

    "against_with_plan": """You are a skilled debater arguing AGAINST the following topic.
Provide well-reasoned, persuasive arguments. Address your opponent's points when relevant.

First, create a strategic plan inside <plan>...</plan> tags:
- 3-6 key points or strategies for your argument
- 1-2 anticipated counterarguments to address

Then, write your persuasive argument inside <debate>...</debate> tags.
Keep your response concise but impactful (3-5 sentences max).""",

    "for_simple": """You are a skilled debater arguing FOR the following topic.
Provide well-reasoned, persuasive arguments. Address your opponent's points when relevant.
Keep your response concise but impactful (3-5 sentences max).""",

    "against_simple": """You are a skilled debater arguing AGAINST the following topic.
Provide well-reasoned, persuasive arguments. Address your opponent's points when relevant.
Keep your response concise but impactful (3-5 sentences max).""",
}


def get_system_prompt(position: str, include_plan: bool = True) -> str:
    """Get system prompt for given position and plan setting."""
    key = f"{position.lower()}_{'with_plan' if include_plan else 'simple'}"
    return SYSTEM_PROMPTS.get(key, SYSTEM_PROMPTS["for_with_plan"])


def build_state_text(
    topic: str,
    debate_history: List[Dict[str, Any]],
    position: str,
    max_turns: Optional[int] = 6,
) -> str:
    """
    Build compact state text for critic/reward computation.

    Similar to original pnlc/text.py build_state_text.
    """
    parts = [
        f"TOPIC: {topic}",
        f"POSITION: {position}",
    ]

    if debate_history:
        parts.append("DEBATE SO FAR:")

        history = debate_history
        if max_turns is not None and len(history) > max_turns * 2:
            history = history[-(max_turns * 2):]

        for entry in history:
            agent = entry.get("agent", "UNKNOWN")
            argument = entry.get("argument", "")
            turn = entry.get("turn", "?")
            parts.append(f"Turn {turn} - {agent}: {argument}")

    return "\n".join(parts)


def build_user_prompt(
    topic: str,
    position: str,
    turn: int,
    debate_history: List[Dict[str, Any]],
    include_plan_instruction: bool = True,
    max_history_turns: Optional[int] = 6,
) -> str:
    """Build user prompt for debate generation."""
    parts = [
        f"Topic: {topic}",
        f"Your position: {position}",
        f"Current turn: {turn}",
    ]

    if debate_history:
        history = debate_history
        if max_history_turns is not None and len(history) > max_history_turns * 2:
            history = history[-(max_history_turns * 2):]

        history_lines = []
        for entry in history:
            agent = entry.get("agent", "UNKNOWN")
            argument = entry.get("argument", "")
            t = entry.get("turn", "?")
            history_lines.append(f"Turn {t} - {agent}: {argument}")

        parts.append(f"\nDEBATE SO FAR:\n" + "\n".join(history_lines))

    prompt = "\n".join(parts)

    if include_plan_instruction:
        prompt += "\n\nFirst create your strategic plan, then your argument."

    return prompt


def extract_plan(text: str) -> str:
    """Extract plan from <plan>...</plan> tags."""
    pattern = r"<plan>(.*?)</plan>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: look for bullet points
    lines = text.strip().split("\n")
    plan_lines = []
    for line in lines:
        if line.strip().startswith(("-", "*", "•")):
            plan_lines.append(line)
        elif plan_lines:
            break

    return "\n".join(plan_lines) if plan_lines else ""


def extract_utterance(text: str) -> str:
    """Extract utterance from <debate>...</debate> tags."""
    pattern = r"<debate>(.*?)</debate>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: remove plan and take rest
    cleaned = re.sub(r"<plan>.*?</plan>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?debate>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def strip_tags(text: str) -> str:
    """Remove all XML-style tags from text."""
    return re.sub(r"<[^>]+>", "", text).strip()


def pseudo_plan_from_argument(argument: str, num_sentences: int = 2) -> str:
    """
    Create pseudo-plan from argument (first N sentences).

    Useful when explicit plans are not available.
    """
    sentences = re.split(r"(?<=[.!?])\s+", argument.strip())
    return " ".join(sentences[:num_sentences])


def format_debate_history(history: List[Dict[str, Any]]) -> str:
    """Format debate history as readable text."""
    lines = []
    for entry in history:
        agent = entry.get("agent", "UNKNOWN")
        argument = entry.get("argument", "")
        turn = entry.get("turn", "?")
        lines.append(f"Turn {turn} - {agent}: {argument}")
    return "\n\n".join(lines)


def count_sentences(text: str) -> int:
    """Count sentences in text."""
    sentences = re.split(r"[.!?]+", text.strip())
    return len([s for s in sentences if s.strip()])


def count_bullets(text: str) -> int:
    """Count bullet points in text."""
    return len(re.findall(r"^[\s]*[-*•]\s", text, re.MULTILINE))


def validate_plan_format(plan: str) -> Dict[str, Any]:
    """Validate plan format and return metrics."""
    bullet_count = count_bullets(plan)
    has_bullets = bullet_count >= 3
    not_too_long = bullet_count <= 6
    char_count = len(plan)
    in_length_range = 50 <= char_count <= 500

    return {
        "valid": has_bullets and not_too_long and in_length_range,
        "bullet_count": bullet_count,
        "char_count": char_count,
        "has_bullets": has_bullets,
        "not_too_long": not_too_long,
        "in_length_range": in_length_range,
    }


def validate_utterance_format(utterance: str) -> Dict[str, Any]:
    """Validate utterance format and return metrics."""
    sentence_count = count_sentences(utterance)
    has_content = len(utterance.strip()) > 20
    in_sentence_range = 3 <= sentence_count <= 6

    return {
        "valid": has_content and in_sentence_range,
        "sentence_count": sentence_count,
        "char_count": len(utterance),
        "has_content": has_content,
        "in_sentence_range": in_sentence_range,
    }


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


class TokenCounter:
    """Utility for counting tokens."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)
