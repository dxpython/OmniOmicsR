#!/usr/bin/env python3
"""
Train OmniGraphDiff Model

Complete training script with data loading, training loop, evaluation, and checkpointing.

Usage:
    python scripts/train_omnigraphdiff.py --config configs/default_tcga.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OmniGraphDiff imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from omnigraphdiff.models import OmniGraphDiffModel
from omnigraphdiff.losses import compute_total_loss


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleMultiOmicsDataset(torch.utils.data.Dataset):
    """
    Simple dataset for multi-omics data.

    For real use, load from h5py/npz files.
    """

    def __init__(self, data_dict, graphs_dict=None, clinical_data=None):
        self.modalities = data_dict
        self.graphs = graphs_dict or {}
        self.clinical_data = clinical_data or {}
        self.n_samples = list(data_dict.values())[0].shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        batch = {
            "modalities": {mod: torch.FloatTensor(data[idx]) for mod, data in self.modalities.items()},
            "targets": {mod: torch.FloatTensor(data[idx]) for mod, data in self.modalities.items()},
        }

        if self.graphs:
            batch["feature_graphs"] = self.graphs.get("feature_graphs", {})
            batch["sample_graph"] = self.graphs.get("sample_graph")

        if self.clinical_data:
            batch["clinical_data"] = {
                k: torch.FloatTensor([v[idx]]) if isinstance(v, np.ndarray) else v[idx]
                for k, v in self.clinical_data.items()
            }

        return batch


def train_epoch(model, dataloader, optimizer, config, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {}

    for batch in tqdm(dataloader, desc="Training"):
        # Move batch to device
        batch = {
            k: {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
            if isinstance(v, dict) else v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(batch)

                # Compute loss
                loss, loss_dict = compute_total_loss(
                    outputs,
                    batch,
                    config["model"],
                    feature_graphs=batch.get("feature_graphs"),
                    sample_graph=batch.get("sample_graph")
                )

            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(batch)
            loss, loss_dict = compute_total_loss(
                outputs,
                batch,
                config["model"],
                feature_graphs=batch.get("feature_graphs"),
                sample_graph=batch.get("sample_graph")
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
            optimizer.step()

        total_loss += loss.item()

        # Accumulate loss components
        for k, v in loss_dict.items():
            if k not in loss_components:
                loss_components[k] = 0.0
            loss_components[k] += v.item() if isinstance(v, torch.Tensor) else v

    # Average losses
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}

    return avg_loss, avg_components


@torch.no_grad()
def validate(model, dataloader, config, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    loss_components = {}

    for batch in tqdm(dataloader, desc="Validation"):
        # Move batch to device
        batch = {
            k: {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
            if isinstance(v, dict) else v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward pass
        outputs = model(batch)

        # Compute loss
        loss, loss_dict = compute_total_loss(
            outputs,
            batch,
            config["model"],
            feature_graphs=batch.get("feature_graphs"),
            sample_graph=batch.get("sample_graph")
        )

        total_loss += loss.item()

        for k, v in loss_dict.items():
            if k not in loss_components:
                loss_components[k] = 0.0
            loss_components[k] += v.item() if isinstance(v, torch.Tensor) else v

    # Average losses
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}

    return avg_loss, avg_components


def main():
    parser = argparse.ArgumentParser(description="Train OmniGraphDiff Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Set seed
    set_seed(config.get("seed", 42))

    # Setup device
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # Initialize model
    logger.info("Initializing model...")
    model = OmniGraphDiffModel(config["model"]).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy data for demonstration
    # In real use, replace with actual data loading
    logger.info("Loading data...")
    n_samples = 1000
    dummy_data = {
        "rna": np.random.randn(n_samples, config["model"]["modalities"]["rna"]["input_dim"]).astype(np.float32),
        "protein": np.random.randn(n_samples, config["model"]["modalities"]["protein"]["input_dim"]).astype(np.float32),
        "cnv": np.random.randn(n_samples, config["model"]["modalities"]["cnv"]["input_dim"]).astype(np.float32),
    }

    dummy_clinical = {
        "survival_time": np.random.exponential(100, n_samples).astype(np.float32),
        "event": np.random.binomial(1, 0.6, n_samples).astype(np.float32),
    }

    # Create datasets
    train_size = int(0.8 * n_samples)
    train_dataset = SimpleMultiOmicsDataset(
        {k: v[:train_size] for k, v in dummy_data.items()},
        clinical_data={k: v[:train_size] for k, v in dummy_clinical.items()}
    )
    val_dataset = SimpleMultiOmicsDataset(
        {k: v[train_size:] for k, v in dummy_data.items()},
        clinical_data={k: v[train_size:] for k, v in dummy_clinical.items()}
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config["training"].get("mixed_precision", False) and device.type == "cuda" else None

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config["training"]["early_stopping"]["patience"]

    logger.info("Starting training...")
    for epoch in range(1, config["training"]["epochs"] + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{config['training']['epochs']}")
        logger.info(f"{'='*50}")

        # Train
        train_loss, train_components = train_epoch(model, train_loader, optimizer, config, device, scaler)
        logger.info(f"Train Loss: {train_loss:.4f}")
        for k, v in train_components.items():
            logger.info(f"  {k}: {v:.4f}")

        # Validate
        if epoch % config["logging"]["val_frequency"] == 0:
            val_loss, val_components = validate(model, val_loader, config, device)
            logger.info(f"Val Loss: {val_loss:.4f}")
            for k, v in val_components.items():
                logger.info(f"  {k}: {v:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"✓ New best validation loss: {best_val_loss:.4f}")

                # Save best model
                checkpoint_dir = Path(config["training"]["checkpoint_dir"])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                }, checkpoint_dir / "best_model.pt")
            else:
                patience_counter += 1
                logger.info(f"Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    logger.info("Early stopping triggered!")
                    break

    logger.info("\nTraining complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
