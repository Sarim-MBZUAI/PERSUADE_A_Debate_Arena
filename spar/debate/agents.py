"""
Debate Agents

FOR and AGAINST agents with plan+utterance generation.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .state import DebateState
from .action import DebateAction, format_plan_prompt, format_utterance_prompt
from ..config import DebateConfig, ModelConfig


class BaseDebateAgent(ABC):
    """Abstract base class for debate agents."""

    def __init__(self, position: str, config: DebateConfig):
        self.position = position
        self.config = config

    @abstractmethod
    def act(self, state: DebateState) -> DebateAction:
        """Generate plan and utterance for current state."""
        pass

    @abstractmethod
    def generate_plan(
        self,
        state: DebateState,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate only the plan."""
        pass

    @abstractmethod
    def generate_utterance(
        self,
        state: DebateState,
        plan: str,
    ) -> str:
        """Generate utterance given plan."""
        pass


class LLMDebateAgent(BaseDebateAgent):
    """
    LLM-based debate agent.

    Can be trainable (for RL) or frozen (for opponent).
    """

    def __init__(
        self,
        position: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: DebateConfig,
        trainable: bool = False,
    ):
        super().__init__(position, config)
        self.model = model
        self.tokenizer = tokenizer
        self.trainable = trainable

        # System prompt for the agent
        self.system_prompt = f"""You are a skilled debater arguing {position} the following topic.
Provide well-reasoned, persuasive arguments. Address your opponent's points when relevant.

First, create a strategic plan inside <plan>...</plan> tags:
- 3-6 key points or strategies for your argument
- 1-2 anticipated counterarguments to address

Then, write your persuasive argument inside <debate>...</debate> tags.
Keep your response concise but impactful (3-5 sentences max)."""

    def act(self, state: DebateState) -> DebateAction:
        """Generate plan and utterance together."""
        state = state.with_position(self.position)
        prompt = self._format_prompt(state)

        # Generate full response (plan + utterance)
        response, log_probs = self._generate(
            prompt,
            max_new_tokens=self.config.max_tokens_per_turn,
            temperature=self.config.plan_temperature,
            do_sample=True,
        )

        # Parse into action
        action = DebateAction.from_raw_text(
            text=response,
            position=self.position,
            turn=state.turn,
            log_probs=log_probs if self.trainable else None,
        )

        return action

    def generate_plan(
        self,
        state: DebateState,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate only the plan (for GRPO sampling)."""
        state = state.with_position(self.position)
        prompt = format_plan_prompt(state.to_state_text())

        # Apply system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.plan_max_tokens,
                temperature=temperature or self.config.plan_temperature,
                top_p=self.config.plan_top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Extract plan from response
        plan = DebateAction._extract_plan(response)
        return plan if plan else response.strip()

    def generate_utterance(
        self,
        state: DebateState,
        plan: str,
    ) -> str:
        """Generate utterance given plan (deterministic)."""
        state = state.with_position(self.position)
        prompt = format_utterance_prompt(state.to_state_text(), plan)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens_per_turn - self.config.plan_max_tokens,
                temperature=self.config.utterance_temperature,
                do_sample=False,  # Greedy for utterance
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Extract utterance from response
        utterance = DebateAction._extract_utterance(response)
        return utterance if utterance else response.strip()

    def generate_multiple_plans(
        self,
        state: DebateState,
        num_plans: int = 4,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate multiple plan candidates for GRPO."""
        plans = []
        for k in range(num_plans):
            plan = self.generate_plan(
                state,
                temperature=temperature,
                seed=self.config.seed + k,
            )
            plans.append(plan)
        return plans

    def _format_prompt(self, state: DebateState) -> str:
        """Format full prompt for generation."""
        user_prompt = state.to_prompt(include_plan_instruction=True)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """Generate response and optionally return log probs."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        if self.trainable:
            # Return log probs for training
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

            # Compute log probs from scores
            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            log_probs = self._compute_log_probs(outputs.scores, generated_ids)

            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response, log_probs
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return response, None

    def _compute_log_probs(
        self,
        scores: Tuple[torch.Tensor, ...],
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities from generation scores."""
        log_probs = []
        for i, score in enumerate(scores):
            if i < len(generated_ids):
                probs = torch.softmax(score[0], dim=-1)
                token_id = generated_ids[i]
                log_prob = torch.log(probs[token_id] + 1e-10)
                log_probs.append(log_prob)

        return torch.stack(log_probs) if log_probs else torch.tensor([])

    def get_log_probs(
        self,
        text: str,
        context: str = "",
        with_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get log probabilities for a given text string.

        Args:
            text: The text to compute log probs for
            context: Optional context/prompt that preceded the text
            with_grad: If True, compute log probs with gradients for training

        Returns:
            - log_probs: Token-level log probabilities
            - token_ids: Token IDs
            - mask: Attention mask
        """
        # Combine context and text
        if context:
            full_text = context + text
        else:
            full_text = text

        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.model.device)

        # Get context length for masking
        if context:
            context_inputs = self.tokenizer(context, return_tensors="pt")
            context_len = context_inputs["input_ids"].shape[1]
        else:
            context_len = 0

        # Forward pass to get logits and compute log probs
        if with_grad and self.trainable:
            # Ensure model is in train mode for gradient computation
            self.model.train()

            # Enable gradient computation for the forward pass
            with torch.enable_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Compute log probs: shift by 1 (predict next token)
                shift_logits = logits[:, :-1, :]
                shift_labels = inputs["input_ids"][:, 1:]

                log_probs_all = torch.log_softmax(shift_logits, dim=-1)

                # Get log prob of actual tokens
                log_probs = torch.gather(
                    log_probs_all,
                    dim=-1,
                    index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                # Create mask for the generated text (not context)
                seq_len = log_probs.shape[1]
                mask = torch.zeros(log_probs.shape, device=log_probs.device)
                if context_len > 0:
                    mask[:, context_len-1:] = 1.0
                else:
                    mask[:, :] = 1.0

            return log_probs.squeeze(0), inputs["input_ids"].squeeze(0), mask.squeeze(0)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Compute log probs: shift by 1 (predict next token)
                shift_logits = logits[:, :-1, :]
                shift_labels = inputs["input_ids"][:, 1:]

                log_probs_all = torch.log_softmax(shift_logits, dim=-1)

                # Get log prob of actual tokens
                log_probs = torch.gather(
                    log_probs_all,
                    dim=-1,
                    index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                # Create mask for the generated text (not context)
                seq_len = log_probs.shape[1]
                mask = torch.zeros_like(log_probs)
                if context_len > 0:
                    mask[:, context_len-1:] = 1.0
                else:
                    mask[:, :] = 1.0

            return log_probs.squeeze(0), inputs["input_ids"].squeeze(0), mask.squeeze(0)


class FrozenAgent(LLMDebateAgent):
    """Frozen agent for asymmetric training (opponent doesn't learn).

    NOTE: This agent does NOT freeze model parameters to avoid breaking
    gradients when the same model is shared with TrainableAgent.
    The model is simply used in no_grad mode for inference.
    """

    def __init__(
        self,
        position: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: DebateConfig,
    ):
        super().__init__(position, model, tokenizer, config, trainable=False)
        # NOTE: We do NOT freeze model parameters here because in self-play
        # the same model object may be shared with TrainableAgent.
        # Freezing would break gradients for training.
        # The trainable=False flag ensures get_log_probs uses torch.no_grad().


class TrainableAgent(LLMDebateAgent):
    """Trainable agent for RL training."""

    def __init__(
        self,
        position: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: DebateConfig,
    ):
        super().__init__(position, model, tokenizer, config, trainable=True)


def create_agents(
    for_model: AutoModelForCausalLM,
    against_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: DebateConfig,
    self_play_mode: str = "asymmetric",
) -> Tuple[BaseDebateAgent, BaseDebateAgent]:
    """
    Create FOR and AGAINST agents based on self-play mode.

    Args:
        for_model: Model for FOR agent
        against_model: Model for AGAINST agent (can be same as for_model)
        tokenizer: Shared tokenizer
        config: Debate configuration
        self_play_mode: "asymmetric" (only FOR trains) or "symmetric" (both train)

    Returns:
        Tuple of (for_agent, against_agent)
    """
    if self_play_mode == "asymmetric":
        for_agent = TrainableAgent("FOR", for_model, tokenizer, config)
        against_agent = FrozenAgent("AGAINST", against_model, tokenizer, config)
    else:  # symmetric
        for_agent = TrainableAgent("FOR", for_model, tokenizer, config)
        against_agent = TrainableAgent("AGAINST", against_model, tokenizer, config)

    return for_agent, against_agent
