#!/usr/bin/env python3
"""
Minimal Synthetic Demo for OmniGraphDiff

Generates synthetic multi-omics data and trains OmniGraphDiff for a few epochs.
Perfect for testing installation and understanding the workflow.

Usage:
    python examples/minimal_synthetic_demo.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omnigraphdiff.models import OmniGraphDiffModel
from omnigraphdiff.losses import compute_total_loss


def generate_synthetic_data(n_samples=200, n_features_rna=500, n_features_protein=300):
    """Generate synthetic multi-omics data."""
    print("Generating synthetic data...")

    # Simulate RNA-seq counts (negative binomial-like)
    rna_data = np.random.negative_binomial(5, 0.3, (n_samples, n_features_rna)).astype(np.float32)

    # Simulate proteomics (Gaussian)
    protein_data = np.random.randn(n_samples, n_features_protein).astype(np.float32)

    # Simulate clinical data
    survival_time = np.random.exponential(100, n_samples).astype(np.float32)
    event = np.random.binomial(1, 0.6, n_samples).astype(np.float32)

    print(f"  RNA-seq: {rna_data.shape}")
    print(f"  Proteomics: {protein_data.shape}")
    print(f"  Clinical: {n_samples} patients")

    return {
        "rna": rna_data,
        "protein": protein_data,
        "survival_time": survival_time,
        "event": event
    }


def create_simple_graph(n_nodes, k=10):
    """Create a simple k-NN graph."""
    # Random features for graph construction
    features = np.random.randn(n_nodes, 50)

    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    dist = cdist(features, features)

    # Create k-NN adjacency
    indices = []
    values = []

    for i in range(n_nodes):
        # Get k nearest neighbors (excluding self)
        neighbors = np.argsort(dist[i])[1:k+1]
        for j in neighbors:
            indices.append([i, j])
            values.append(1.0)

    indices = torch.LongTensor(indices).t()
    values = torch.FloatTensor(values)

    # Create sparse tensor
    graph = torch.sparse_coo_tensor(
        indices,
        values,
        (n_nodes, n_nodes)
    )

    return graph


def main():
    print("="*60)
    print("OmniGraphDiff Minimal Synthetic Demo")
    print("="*60)

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate data
    data = generate_synthetic_data(n_samples=200, n_features_rna=500, n_features_protein=300)

    # Create configuration
    config = {
        "latent_dim_shared": 16,
        "use_cross_attention": True,
        "use_sample_gnn": False,  # Disable for simplicity

        "modalities": {
            "rna": {
                "input_dim": 500,
                "latent_dim_specific": 8,
                "encoder_hidden_dims": [256, 128],
                "decoder_hidden_dims": [128, 256],
                "output_dist": "gaussian",  # Simplified (normally "nb")
                "use_feature_gnn": False,
            },
            "protein": {
                "input_dim": 300,
                "latent_dim_specific": 8,
                "encoder_hidden_dims": [128, 64],
                "decoder_hidden_dims": [64, 128],
                "output_dist": "gaussian",
                "use_feature_gnn": False,
            },
        },

        "use_clinical_head": True,
        "use_survival": True,
        "use_classification": False,
        "clinical_hidden_dims": [32],

        # Loss weights
        "loss_weights": {
            "kl": 0.1,
            "graph": 0.0,  # No graph regularization for this demo
            "clinical": 1.0,
            "contrastive": 0.1,
        },

        "use_feature_graph_reg": False,
        "use_sample_graph_reg": False,
        "use_contrastive": True,
    }

    # Initialize model
    print("\nInitializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = OmniGraphDiffModel(config).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create simple dataset
    batch = {
        "modalities": {
            "rna": torch.FloatTensor(data["rna"]).to(device),
            "protein": torch.FloatTensor(data["protein"]).to(device),
        },
        "targets": {
            "rna": torch.FloatTensor(data["rna"]).to(device),
            "protein": torch.FloatTensor(data["protein"]).to(device),
        },
        "clinical_data": {
            "survival_time": torch.FloatTensor(data["survival_time"]).to(device),
            "event": torch.FloatTensor(data["event"]).to(device),
        },
    }

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("\nTraining for 5 epochs...")
    model.train()

    for epoch in range(1, 6):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Compute loss
        loss, loss_dict = compute_total_loss(
            outputs,
            batch,
            config
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print progress
        print(f"\nEpoch {epoch}/5:")
        print(f"  Total Loss: {loss.item():.4f}")
        print(f"  Recon Loss: {loss_dict.get('recon', 0):.4f}")
        print(f"  KL Loss: {loss_dict.get('kl', 0):.4f}")
        print(f"  Clinical Loss: {loss_dict.get('clinical', 0):.4f}")
        print(f"  Contrastive Loss: {loss_dict.get('contrastive', 0):.4f}")

    # Extract latent embeddings
    print("\nExtracting latent embeddings...")
    model.eval()
    with torch.no_grad():
        latents = model.encode(batch)

    print(f"  Shared latent shape: {latents['shared'].shape}")
    for mod in latents['specific'].keys():
        print(f"  {mod} specific latent shape: {latents['specific'][mod].shape}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Modify config for your own data")
    print("  2. Use scripts/train_omnigraphdiff.py for full training")
    print("  3. See configs/default_tcga.yaml for complete configuration")


if __name__ == "__main__":
    main()
