"""SPAR Training Module.

Components:
- SFTTrainer: supervised fine-tuning on teacher demonstrations
- OfflineGRPOTrainer: GRPO over logged arena debates
- HybridTrainer: online GRPO with live student generation and dataset opponents
- DPOTrainer: direct preference optimization on arena preference pairs
"""

from .sft_trainer import SFTConfig, SFTTrainer, DebateSFTDataset
from .offline_grpo_trainer import OfflineGRPOConfig, OfflineGRPOTrainer
from .hybrid_trainer import HybridTrainingConfig, HybridTrainer
from .dpo_trainer import DPOConfig, DPOTrainer, PreferencePairDataset

__all__ = [
    # SFT
    "SFTConfig",
    "SFTTrainer",
    "DebateSFTDataset",
    # Offline GRPO
    "OfflineGRPOConfig",
    "OfflineGRPOTrainer",
    # Hybrid
    "HybridTrainingConfig",
    "HybridTrainer",
    # DPO
    "DPOConfig",
    "DPOTrainer",
    "PreferencePairDataset",
]
