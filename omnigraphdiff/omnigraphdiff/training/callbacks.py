"""
Training Callbacks
"""

import torch
from pathlib import Path
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class."""

    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict
    ) -> bool:
        """
        Called at the end of each epoch.

        Returns:
            bool: Whether to stop training
        """
        return False


class EarlyStopping(Callback):
    """Early stopping callback."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        metric: str = "total",
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode

        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.counter = 0

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        metrics = val_metrics if val_metrics else train_metrics
        current_score = metrics.get(self.metric, metrics.get("total"))

        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(f"Early stopping: no improvement for {self.patience} epochs")
            return True

        return False


class ModelCheckpoint(Callback):
    """Model checkpointing callback."""

    def __init__(
        self,
        filepath: str,
        monitor: str = "total",
        mode: str = "min",
        save_best_only: bool = True,
        save_frequency: int = 1
    ):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency

        self.best_score = float('inf') if mode == "min" else float('-inf')

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        metrics = val_metrics if val_metrics else train_metrics
        current_score = metrics.get(self.monitor, metrics.get("total"))

        should_save = False

        if self.save_best_only:
            if self.mode == "min":
                improved = current_score < self.best_score
            else:
                improved = current_score > self.best_score

            if improved:
                self.best_score = current_score
                should_save = True
                logger.info(f"✓ New best {self.monitor}: {current_score:.4f}")
        else:
            should_save = (epoch % self.save_frequency == 0)

        if should_save:
            filepath = str(self.filepath).format(epoch=epoch, score=current_score)
            trainer.save_checkpoint(filepath)

        return False


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metrics = val_metrics if val_metrics else train_metrics
            self.scheduler.step(metrics.get("total"))
        else:
            self.scheduler.step()

        return False
