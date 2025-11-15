"""
Complete Trainer Class with Multi-GPU, Mixed Precision, and Callbacks
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Optional, List, Callable
from datetime import datetime
import json

from ..losses import compute_total_loss
from .callbacks import Callback


logger = logging.getLogger(__name__)


class Trainer:
    """
    Complete training infrastructure with all production features.

    Features:
    - Multi-GPU training (DDP)
    - Mixed precision training
    - Gradient accumulation
    - Callbacks (early stopping, checkpointing, LR scheduling)
    - TensorBoard logging
    - Comprehensive metrics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        config: Dict,
        device: torch.device,
        callbacks: Optional[List[Callback]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        distributed: bool = False,
        local_rank: int = 0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.callbacks = callbacks or []
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.distributed = distributed
        self.local_rank = local_rank

        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None

        # Directories
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            logger.warning("TensorBoard not available")
            self.writer = None

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        # Setup distributed if needed
        if distributed:
            self.model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )

        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Mixed Precision: {use_amp}")
        logger.info(f"  Distributed: {distributed}")
        logger.info(f"  Gradient Accumulation: {gradient_accumulation_steps}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = len(self.train_loader)

        # Progress bar
        pbar = tqdm(
            enumerate(self.train_loader),
            total=num_batches,
            desc=f"Epoch {self.current_epoch}",
            disable=self.distributed and self.local_rank != 0
        )

        for batch_idx, batch in pbar:
            # Move batch to device
            batch = self._batch_to_device(batch)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(batch)
                    loss, loss_dict = compute_total_loss(
                        outputs,
                        batch,
                        self.config,
                        feature_graphs=batch.get("feature_graphs"),
                        sample_graph=batch.get("sample_graph")
                    )
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass with scaling
                self.scaler.scale(loss).backward()

                # Update weights
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

            else:
                # Standard training
                outputs = self.model(batch)
                loss, loss_dict = compute_total_loss(
                    outputs,
                    batch,
                    self.config,
                    feature_graphs=batch.get("feature_graphs"),
                    sample_graph=batch.get("sample_graph")
                )
                loss = loss / self.gradient_accumulation_steps

                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Accumulate losses
            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += val

            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item() * self.gradient_accumulation_steps,
                "lr": self.optimizer.param_groups[0]["lr"]
            })

            # Log to TensorBoard
            if self.writer and batch_idx % 10 == 0:
                self.writer.add_scalar("train/batch_loss", loss.item(), self.global_step)

            self.global_step += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        return avg_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        epoch_losses = {}
        num_batches = len(self.val_loader)

        for batch in tqdm(
            self.val_loader,
            desc="Validation",
            disable=self.distributed and self.local_rank != 0
        ):
            batch = self._batch_to_device(batch)

            # Forward pass
            outputs = self.model(batch)
            loss, loss_dict = compute_total_loss(
                outputs,
                batch,
                self.config,
                feature_graphs=batch.get("feature_graphs"),
                sample_graph=batch.get("sample_graph")
            )

            # Accumulate losses
            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += val

        # Average losses
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        return avg_losses

    def fit(self, num_epochs: int):
        """Main training loop."""
        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate() if self.val_loader else {}

            # Learning rate step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("total", train_metrics["total"]))
                else:
                    self.scheduler.step()

            # Log metrics
            self._log_metrics(train_metrics, val_metrics)

            # Update history
            self.history["train_loss"].append(train_metrics["total"])
            if val_metrics:
                self.history["val_loss"].append(val_metrics["total"])
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            # Callbacks
            stop_training = False
            for callback in self.callbacks:
                if callback.on_epoch_end(self, epoch, train_metrics, val_metrics):
                    stop_training = True
                    break

            if stop_training:
                logger.info("Early stopping triggered!")
                break

        logger.info("Training complete!")

        # Close TensorBoard
        if self.writer:
            self.writer.close()

        return self.history

    def save_checkpoint(
        self,
        filepath: str,
        include_optimizer: bool = True,
        additional_state: Optional[Dict] = None
    ):
        """Save model checkpoint."""
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config,
        }

        if include_optimizer:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler:
                state["scheduler_state_dict"] = self.scheduler.state_dict()
            if self.scaler:
                state["scaler_state_dict"] = self.scaler.state_dict()

        if additional_state:
            state.update(additional_state)

        torch.save(state, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self.history = checkpoint.get("history", {})

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if self.scaler and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Checkpoint loaded: {filepath}")
        logger.info(f"  Resuming from epoch {self.current_epoch}")

    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device recursively."""
        if isinstance(batch, dict):
            return {
                k: self._batch_to_device(v) for k, v in batch.items()
            }
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, list):
            return [self._batch_to_device(item) for item in batch]
        else:
            return batch

    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to console and TensorBoard."""
        # Console logging
        log_str = f"\nEpoch {self.current_epoch}:"
        log_str += f"\n  Train Loss: {train_metrics['total']:.4f}"
        if val_metrics:
            log_str += f"\n  Val Loss: {val_metrics['total']:.4f}"

        for k, v in train_metrics.items():
            if k != "total":
                log_str += f"\n    train/{k}: {v:.4f}"

        for k, v in val_metrics.items():
            if k != "total":
                log_str += f"\n    val/{k}: {v:.4f}"

        logger.info(log_str)

        # TensorBoard logging
        if self.writer:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, self.current_epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, self.current_epoch)
            self.writer.add_scalar(
                "learning_rate",
                self.optimizer.param_groups[0]["lr"],
                self.current_epoch
            )
