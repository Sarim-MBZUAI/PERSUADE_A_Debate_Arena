"""
Judge Interface

Configurable judge for debate evaluation.
Supports both OpenAI API (GPT-4, GPT-4o) and local LLM judges (Prometheus-2, Llama, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os
import re

from ..config import JudgeConfig


@dataclass
class JudgeEvaluation:
    """Result of judge evaluation."""

    decision: str  # "FOR" or "AGAINST"
    confidence: Optional[float] = None  # 0-1 confidence score
    belief: Optional[float] = None  # 0-1 probability FOR is correct
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None

    @property
    def is_for(self) -> bool:
        return self.decision.upper() == "FOR"

    @property
    def is_against(self) -> bool:
        return self.decision.upper() == "AGAINST"


class JudgeInterface(ABC):
    """Abstract interface for debate judges."""

    @abstractmethod
    def get_initial_position(self, topic: str) -> JudgeEvaluation:
        """Get judge's position before seeing any debate."""
        pass

    @abstractmethod
    def get_final_decision(
        self,
        topic: str,
        debate_history: List[Dict[str, Any]],
    ) -> JudgeEvaluation:
        """Get judge's decision after full debate."""
        pass

    @abstractmethod
    def get_intermediate_belief(
        self,
        topic: str,
        debate_history: List[Dict[str, Any]],
    ) -> JudgeEvaluation:
        """Get judge's current belief probability (for R_bel reward)."""
        pass


class OpenAIJudge(JudgeInterface):
    """
    GPT-4/GPT-4o judge implementation.

    Supports configurable models for different queries:
    - Main model for initial position and final decision
    - Cheaper model for intermediate belief queries
    """

    def __init__(self, config: JudgeConfig):
        from openai import OpenAI
        self.config = config
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Prompts
        self.initial_position_prompt = """You are a judge evaluating the following debate topic:

Topic: {topic}

Before the debate begins, what is your initial position on this topic?
You must choose either FOR or AGAINST (no neutral position allowed).

Respond with ONLY ONE WORD: FOR or AGAINST"""

        self.final_decision_prompt = """You are a judge evaluating a debate on the following topic:

Topic: {topic}

DEBATE TRANSCRIPT:
{debate_text}

After reviewing the full debate, you must make a final decision.
You MUST choose either FOR or AGAINST (no neutral position allowed).
Which side presented the stronger arguments?

Respond with ONLY ONE WORD: FOR or AGAINST"""

        self.belief_query_prompt = """You are evaluating an ongoing debate on the following topic:

Topic: {topic}

DEBATE SO FAR:
{debate_text}

Based on the arguments presented so far, estimate the probability (0-100%) that the FOR side is currently winning.

Respond with ONLY a number between 0 and 100."""

    def get_initial_position(self, topic: str) -> JudgeEvaluation:
        """Get judge's initial position before debate."""
        prompt = self.initial_position_prompt.format(topic=topic)

        response = self._call_api(
            prompt,
            model=self.config.model,
            max_tokens=10,
        )

        decision = self._parse_decision(response)

        return JudgeEvaluation(
            decision=decision,
            raw_response=response,
        )

    def get_final_decision(
        self,
        topic: str,
        debate_history: List[Dict[str, Any]],
    ) -> JudgeEvaluation:
        """Get judge's final decision after debate."""
        debate_text = self._format_debate(debate_history)
        prompt = self.final_decision_prompt.format(
            topic=topic,
            debate_text=debate_text,
        )

        response = self._call_api(
            prompt,
            model=self.config.model,
            max_tokens=10,
        )

        decision = self._parse_decision(response)

        return JudgeEvaluation(
            decision=decision,
            raw_response=response,
        )

    def get_intermediate_belief(
        self,
        topic: str,
        debate_history: List[Dict[str, Any]],
    ) -> JudgeEvaluation:
        """Get judge's current belief probability."""
        debate_text = self._format_debate(debate_history)
        prompt = self.belief_query_prompt.format(
            topic=topic,
            debate_text=debate_text,
        )

        # Use cheaper model for intermediate queries
        model = self.config.intermediate_model or self.config.model

        response = self._call_api(
            prompt,
            model=model,
            max_tokens=10,
        )

        belief = self._parse_belief(response)

        # Convert belief to decision
        decision = "FOR" if belief >= 0.5 else "AGAINST"

        return JudgeEvaluation(
            decision=decision,
            belief=belief,
            raw_response=response,
        )

    def _call_api(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
    ) -> str:
        """Make API call to OpenAI."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=max_tokens,
            seed=self.config.seed,
        )
        return response.choices[0].message.content.strip()

    def _parse_decision(self, response: str) -> str:
        """Parse FOR/AGAINST decision from response."""
        response_upper = response.upper().strip()

        if "FOR" in response_upper and "AGAINST" not in response_upper:
            return "FOR"
        elif "AGAINST" in response_upper:
            return "AGAINST"
        else:
            # Fallback: take first word
            first_word = response_upper.split()[0] if response_upper else "FOR"
            if first_word in ("FOR", "AGAINST"):
                return first_word
            return "FOR"  # Default

    def _parse_belief(self, response: str) -> float:
        """Parse belief probability from response."""
        try:
            # Extract number from response
            numbers = re.findall(r"\d+(?:\.\d+)?", response)
            if numbers:
                value = float(numbers[0])
                # Normalize to 0-1
                if value > 1:
                    value = value / 100.0
                return max(0.0, min(1.0, value))
        except (ValueError, IndexError):
            pass

        return 0.5  # Default to neutral

    def _format_debate(self, history: List[Dict[str, Any]]) -> str:
        """Format debate history as text."""
        lines = []
        for entry in history:
            agent = entry.get("agent", "UNKNOWN")
            argument = entry.get("argument", "")
            turn = entry.get("turn", "?")
            lines.append(f"Turn {turn} - {agent}: {argument}")
        return "\n\n".join(lines)


class LocalLLMJudge(JudgeInterface):
    """
    Local LLM judge implementation using Hugging Face models.

    Supports Prometheus-2, Llama, Qwen, and other instruction-tuned models.
    Can use 4-bit quantization for larger models (e.g., 70B on 48GB GPU).
    """

    def __init__(self, config: JudgeConfig, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = device or config.local_device or "cuda:0"

        print(f"Loading local judge model: {config.local_model_path}")

        # Determine dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.local_torch_dtype, torch.bfloat16)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.local_model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optional 4-bit quantization
        if config.use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.local_model_path,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.local_model_path,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
            )

        self.model.eval()
        print(f"✓ Local judge loaded on {self.device}")

        # Prompts (same as OpenAI but may need adjustment for local models)
        self.initial_position_prompt = """You are a judge evaluating the following debate topic:

Topic: {topic}

Before the debate begins, what is your initial position on this topic?
You must choose either FOR or AGAINST (no neutral position allowed).

Respond with ONLY ONE WORD: FOR or AGAINST"""

        self.final_decision_prompt = """You are a judge evaluating a debate on the following topic:

Topic: {topic}

DEBATE TRANSCRIPT:
{debate_text}

After reviewing the full debate, you must make a final decision.
You MUST choose either FOR or AGAINST (no neutral position allowed).
Which side presented the stronger arguments?

Respond with ONLY ONE WORD: FOR or AGAINST"""

        self.belief_query_prompt = """You are evaluating an ongoing debate on the following topic:

Topic: {topic}

DEBATE SO FAR:
{debate_text}

Based on the arguments presented so far, estimate the probability (0-100%) that the FOR side is currently winning.

Respond with ONLY a number between 0 and 100."""

    def get_initial_position(self, topic: str) -> JudgeEvaluation:
        """Get judge's initial position before debate."""
        prompt = self.initial_position_prompt.format(topic=topic)
        response = self._generate(prompt, max_new_tokens=10)
        decision = self._parse_decision(response)

        return JudgeEvaluation(
            decision=decision,
            raw_response=response,
        )

    def get_final_decision(
        self,
        topic: str,
        debate_history: List[Dict[str, Any]],
    ) -> JudgeEvaluation:
        """Get judge's final decision after debate."""
        debate_text = self._format_debate(debate_history)
        prompt = self.final_decision_prompt.format(
            topic=topic,
            debate_text=debate_text,
        )

        response = self._generate(prompt, max_new_tokens=10)
        decision = self._parse_decision(response)

        return JudgeEvaluation(
            decision=decision,
            raw_response=response,
        )

    def get_intermediate_belief(
        self,
        topic: str,
        debate_history: List[Dict[str, Any]],
    ) -> JudgeEvaluation:
        """Get judge's current belief probability."""
        debate_text = self._format_debate(debate_history)
        prompt = self.belief_query_prompt.format(
            topic=topic,
            debate_text=debate_text,
        )

        response = self._generate(prompt, max_new_tokens=10)
        belief = self._parse_belief(response)

        # Convert belief to decision
        decision = "FOR" if belief >= 0.5 else "AGAINST"

        return JudgeEvaluation(
            decision=decision,
            belief=belief,
            raw_response=response,
        )

    def _generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        """Generate response from local model."""
        import torch

        # Format as chat message for instruction-tuned models
        messages = [{"role": "user", "content": prompt}]

        # Use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def _parse_decision(self, response: str) -> str:
        """Parse FOR/AGAINST decision from response."""
        response_upper = response.upper().strip()

        if "FOR" in response_upper and "AGAINST" not in response_upper:
            return "FOR"
        elif "AGAINST" in response_upper:
            return "AGAINST"
        else:
            # Fallback: take first word
            first_word = response_upper.split()[0] if response_upper else "FOR"
            if first_word in ("FOR", "AGAINST"):
                return first_word
            return "FOR"  # Default

    def _parse_belief(self, response: str) -> float:
        """Parse belief probability from response."""
        try:
            # Extract number from response
            numbers = re.findall(r"\d+(?:\.\d+)?", response)
            if numbers:
                value = float(numbers[0])
                # Normalize to 0-1
                if value > 1:
                    value = value / 100.0
                return max(0.0, min(1.0, value))
        except (ValueError, IndexError):
            pass

        return 0.5  # Default to neutral

    def _format_debate(self, history: List[Dict[str, Any]]) -> str:
        """Format debate history as text."""
        lines = []
        for entry in history:
            agent = entry.get("agent", "UNKNOWN")
            argument = entry.get("argument", "")
            turn = entry.get("turn", "?")
            lines.append(f"Turn {turn} - {agent}: {argument}")
        return "\n\n".join(lines)


class BatchJudge:
    """
    Wrapper for batch processing multiple judge queries.

    Useful for efficient API calls during training.
    """

    def __init__(self, judge: JudgeInterface, batch_size: int = 8):
        self.judge = judge
        self.batch_size = batch_size

    def get_initial_positions_batch(
        self,
        topics: List[str],
    ) -> List[JudgeEvaluation]:
        """Get initial positions for multiple topics."""
        results = []
        for topic in topics:
            result = self.judge.get_initial_position(topic)
            results.append(result)
        return results

    def get_final_decisions_batch(
        self,
        debates: List[Dict[str, Any]],
    ) -> List[JudgeEvaluation]:
        """Get final decisions for multiple debates."""
        results = []
        for debate in debates:
            result = self.judge.get_final_decision(
                topic=debate["topic"],
                debate_history=debate["history"],
            )
            results.append(result)
        return results

    def get_beliefs_batch(
        self,
        debates: List[Dict[str, Any]],
    ) -> List[JudgeEvaluation]:
        """Get intermediate beliefs for multiple debates."""
        results = []
        for debate in debates:
            result = self.judge.get_intermediate_belief(
                topic=debate["topic"],
                debate_history=debate["history"],
            )
            results.append(result)
        return results


def create_judge(config: JudgeConfig, device: Optional[str] = None) -> JudgeInterface:
    """Factory function to create judge based on config.

    Args:
        config: JudgeConfig specifying judge type and parameters
        device: Optional device override for local judges

    Returns:
        JudgeInterface implementation (OpenAI or Local LLM)
    """
    if config.judge_type == "local":
        return LocalLLMJudge(config, device=device)
    else:
        return OpenAIJudge(config)
