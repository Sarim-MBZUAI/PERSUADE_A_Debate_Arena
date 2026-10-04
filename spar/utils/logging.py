"""
Logging Utilities

Weights & Biases integration and structured logging.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(
    output_dir: Path,
    name: str = "spar",
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup file and console logging."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


class WandbLogger:
    """Weights & Biases logging wrapper."""

    def __init__(
        self,
        project: str = "spar",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and WANDB_AVAILABLE

        if self.enabled:
            wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
            )
            self.run = wandb.run
        else:
            self.run = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb."""
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_table(self, name: str, data: Dict[str, list]) -> None:
        """Log table data to wandb."""
        if self.enabled:
            table = wandb.Table(columns=list(data.keys()))
            rows = zip(*data.values())
            for row in rows:
                table.add_data(*row)
            wandb.log({name: table})

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        """Log summary metrics."""
        if self.enabled:
            for key, value in metrics.items():
                wandb.run.summary[key] = value

    def watch(self, model, log: str = "gradients", log_freq: int = 100) -> None:
        """Watch model gradients/parameters."""
        if self.enabled:
            wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        """Finish wandb run."""
        if self.enabled:
            wandb.finish()


class MetricsTracker:
    """Track and aggregate training metrics."""

    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.step = 0

    def update(self, metrics: Dict[str, float]) -> None:
        """Add metrics for current step."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        self.step += 1

    def get_average(self, key: str, window: int = 100) -> float:
        """Get moving average of metric."""
        if key not in self.metrics:
            return 0.0
        values = self.metrics[key][-window:]
        return sum(values) / len(values) if values else 0.0

    def get_latest(self, key: str) -> float:
        """Get latest value of metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]

    def get_summary(self, window: int = 100) -> Dict[str, float]:
        """Get summary of all metrics."""
        return {
            key: self.get_average(key, window)
            for key in self.metrics
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {}
        self.step = 0


# Global wandb logger instance
_wandb_logger: Optional[WandbLogger] = None


def setup_wandb(
    project: str = "spar",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    entity: Optional[str] = None,
) -> Optional[WandbLogger]:
    """Setup and initialize wandb logging.

    Args:
        project: W&B project name
        name: Run name
        config: Configuration dict to log
        entity: W&B entity (team/user)

    Returns:
        WandbLogger instance or None if wandb not available
    """
    global _wandb_logger

    _wandb_logger = WandbLogger(
        project=project,
        entity=entity,
        name=name,
        config=config,
        enabled=True,
    )

    return _wandb_logger


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
) -> None:
    """Log metrics to wandb if available.

    Args:
        metrics: Dict of metric name to value
        step: Optional step number
    """
    global _wandb_logger

    if _wandb_logger is not None:
        _wandb_logger.log(metrics, step=step)


def finish_wandb() -> None:
    """Finish wandb run."""
    global _wandb_logger

    if _wandb_logger is not None:
        _wandb_logger.finish()
        _wandb_logger = None
