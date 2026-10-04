"""
SPAR Configuration System

Hierarchical configuration for Strategic Planning and Argumentative Reinforcement.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple, Dict, Any
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    """Model and LoRA configuration."""

    name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    torch_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True

    def get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from string."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)


@dataclass
class DebateConfig:
    """Debate generation configuration."""

    num_turns: int = 4
    positions: List[str] = field(default_factory=lambda: ["FOR", "AGAINST"])
    max_tokens_per_turn: int = 512
    plan_max_tokens: int = 192
    history_max_turns: int = 6

    # Generation settings
    plan_temperature: float = 0.7
    plan_top_p: float = 0.95
    utterance_temperature: float = 0.0  # Greedy for utterance
    utterance_top_p: float = 1.0

    # Seed for reproducibility
    seed: int = 42


@dataclass
class JudgeConfig:
    """Judge model configuration (configurable per experiment)."""

    # Judge type: "openai" for API-based, "local" for local LLM
    judge_type: Literal["openai", "local"] = "local"

    # OpenAI settings (used when judge_type="openai")
    model: str = "gpt-4"  # gpt-4, gpt-4o, gpt-4-turbo, gpt-3.5-turbo
    intermediate_model: Optional[str] = "gpt-3.5-turbo"  # Cheaper model for intermediate queries

    # Local LLM judge settings (used when judge_type="local")
    # Best options:
    #   - "AtlaAI/Selene-1-Mini-Llama-3.1-8B" (best 8B, beats GPT-4o on benchmarks)
    #   - "prometheus-eval/prometheus-7b-v2.0" (solid baseline, 72-85% human agreement)
    #   - "root-signals/RootSignals-Judge-Llama-70B" (SOTA, needs use_4bit=True)
    local_model_path: str = "AtlaAI/Selene-1-Mini-Llama-3.1-8B"
    local_device: Optional[str] = None  # Device for local judge, auto-detected if None
    local_torch_dtype: str = "bfloat16"  # bfloat16, float16
    use_4bit: bool = False  # Use 4-bit quantization for larger models (e.g., 70B)

    # Common settings
    temperature: float = 0.0
    max_tokens: int = 10
    seed: int = 42
    belief_query_interval: int = 2  # Query every N turns
    belief_scale: int = 100  # Scale for belief queries (e.g., 0-100)

    # Batch settings for efficiency
    batch_size: int = 8
    timeout: float = 30.0


@dataclass
class RewardConfig:
    """Multi-component reward configuration."""

    # Component weights
    outcome_weight: float = 1.0      # R_win: judge verdict
    critic_weight: float = 0.3       # R_crit: V(s,p) potential shaping
    belief_weight: float = 0.2       # R_bel: judge belief dynamics
    diversity_weight: float = 0.1    # R_div: plan diversity
    format_weight: float = 0.05      # R_fmt: format adherence

    # GDPO-style per-component normalization
    normalize_per_component: bool = True
    normalization_eps: float = 1e-8

    # Asymmetric weighting (FOR bonus to overcome AGAINST advantage)
    for_win_bonus: float = 1.5

    # Critic settings
    critic_checkpoint: Optional[Path] = None
    critic_gamma: float = 0.99  # Discount for potential shaping

    # Diversity settings
    diversity_embedding_dim: int = 256
    diversity_temperature: float = 1.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Basic training
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # GRPO settings
    num_generations: int = 4  # K candidates per prompt (group size)
    kl_coef: float = 0.1      # KL penalty coefficient
    clip_epsilon: float = 0.2  # PPO-style clipping
    epsilon_high: Optional[float] = None  # Optional upper bound for positive advantages

    # SAPO-specific (if using SAPO loss)
    sapo_temperature: float = 1.0
    sapo_adaptive_temp: bool = True
    temperature: Optional[float] = None  # SAPO sigmoid temperature (YAML-compatible)
    gate_center: Optional[float] = None  # SAPO gate center (YAML-compatible)

    # Token weighting for hierarchical actions
    plan_token_weight: float = 1.5
    utterance_token_weight: float = 1.0

    # Self-play configuration
    self_play_mode: Literal["asymmetric", "symmetric"] = "asymmetric"

    # Debate randomization
    randomize_roles: bool = False
    randomize_start: bool = False

    # Curriculum learning
    curriculum_mode: Literal["fixed", "progressive"] = "fixed"
    curriculum_schedule: Optional[List[Tuple[int, int]]] = None
    # e.g., [(0, 1), (500, 2), (1000, 3), (1500, 4)]

    # Checkpointing
    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: int = 200
    logging_steps: int = 10

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Loss-specific configuration (passed through to loss functions)
    importance_sampling_level: Literal["token", "sequence"] = "token"
    use_running_stats: bool = True
    running_decay: float = 0.99
    component_weights: Optional[Dict[str, float]] = None
    plan_weight: Optional[float] = None
    utterance_weight: Optional[float] = None
    for_advantage_bonus: Optional[float] = None
    position: Optional[str] = None
    gamma: Optional[float] = None
    normalize_per_component: Optional[bool] = None


@dataclass
class SPARConfig:
    """Master configuration for SPAR training."""

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Loss function
    loss_type: Literal["grpo", "sapo", "gdpo", "spar", "dpo"] = "gdpo"

    # Optional nested configs (preserve YAML blocks)
    curriculum: Optional[Dict[str, Any]] = None
    self_play: Optional[Dict[str, Any]] = None

    # Experiment settings
    experiment_name: str = "spar_default"
    output_dir: Path = field(default_factory=lambda: Path("outputs/default"))

    # Data
    train_topics_file: Path = field(default_factory=lambda: Path("data/all_topics.csv"))
    eval_topics_file: Optional[Path] = None
    train_split_ratio: float = 0.8  # 80% train, 20% eval
    arena_data_dir: Optional[str] = None

    # Training paradigm
    paradigm: Optional[str] = None  # "sft", "offline_grpo", "hybrid", "dpo"

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda:0"
    ref_device: Optional[str] = None  # Device for reference model (if different)

    # Logging
    use_wandb: bool = True
    wandb_project: str = "spar"
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.train_topics_file, str):
            self.train_topics_file = Path(self.train_topics_file)
        if isinstance(self.eval_topics_file, str):
            self.eval_topics_file = Path(self.eval_topics_file)
        if isinstance(self.reward.critic_checkpoint, str):
            self.reward.critic_checkpoint = Path(self.reward.critic_checkpoint)

    @classmethod
    def from_yaml(cls, path: Path) -> "SPARConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "SPARConfig":
        """Recursively construct config from dict."""
        from dataclasses import fields

        def filter_fields(dc_class, dc_data):
            """Filter dict to only include valid dataclass fields."""
            valid_fields = {f.name for f in fields(dc_class)}
            return {k: v for k, v in dc_data.items() if k in valid_fields}

        # Extract sub-configs
        model_data = data.pop("model", {})
        debate_data = data.pop("debate", {})
        judge_data = data.pop("judge", {})
        reward_data = data.pop("reward", {})
        training_data = data.pop("training", {})
        curriculum_data = data.pop("curriculum", None)
        self_play_data = data.pop("self_play", None)

        # Map YAML training aliases to canonical fields
        if "num_trajectories" in training_data and "num_generations" not in training_data:
            training_data["num_generations"] = training_data.pop("num_trajectories")
        if "beta" in training_data and "kl_coef" not in training_data:
            training_data["kl_coef"] = training_data.pop("beta")
        if "epsilon" in training_data and "clip_epsilon" not in training_data:
            training_data["clip_epsilon"] = training_data.pop("epsilon")
        if "log_every" in training_data and "logging_steps" not in training_data:
            training_data["logging_steps"] = training_data.pop("log_every")
        if "eval_every" in training_data and "eval_steps" not in training_data:
            training_data["eval_steps"] = training_data.pop("eval_every")
        if "save_every" in training_data and "save_steps" not in training_data:
            training_data["save_steps"] = training_data.pop("save_every")
        if "use_amp" in training_data:
            use_amp = bool(training_data.pop("use_amp"))
            if "bf16" not in training_data and "fp16" not in training_data:
                training_data["bf16"] = use_amp

        # Map YAML plan/utterance weights to canonical fields
        if "plan_weight" in training_data and "plan_token_weight" not in training_data:
            training_data["plan_token_weight"] = training_data["plan_weight"]
        if "utterance_weight" in training_data and "utterance_token_weight" not in training_data:
            training_data["utterance_token_weight"] = training_data["utterance_weight"]

        # Promote training.use_wandb to top-level use_wandb if set
        if "use_wandb" in training_data and "use_wandb" not in data:
            data["use_wandb"] = training_data["use_wandb"]

        # Map self_play.mode into training.self_play_mode if provided
        if isinstance(self_play_data, dict) and "mode" in self_play_data:
            training_data.setdefault("self_play_mode", self_play_data["mode"])

        # Map curriculum num_turns -> max_turns for scheduler compatibility
        if isinstance(curriculum_data, dict) and "num_turns" in curriculum_data and "max_turns" not in curriculum_data:
            curriculum_data["max_turns"] = curriculum_data["num_turns"]

        # Remove extra keys not in SPARConfig
        valid_spar_fields = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_spar_fields}

        return cls(
            model=ModelConfig(**filter_fields(ModelConfig, model_data)),
            debate=DebateConfig(**filter_fields(DebateConfig, debate_data)),
            judge=JudgeConfig(**filter_fields(JudgeConfig, judge_data)),
            reward=RewardConfig(**filter_fields(RewardConfig, reward_data)),
            training=TrainingConfig(**filter_fields(TrainingConfig, training_data)),
            curriculum=curriculum_data,
            self_play=self_play_data,
            **data
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict

        data = asdict(self)
        # Convert Path objects to strings
        self._paths_to_str(data)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _paths_to_str(self, d: dict) -> None:
        """Recursively convert Path objects to strings."""
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                self._paths_to_str(v)


def load_config(path: str) -> SPARConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        SPARConfig instance
    """
    return SPARConfig.from_yaml(Path(path))
