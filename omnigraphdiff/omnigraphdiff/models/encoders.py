"""
Modality-Specific Encoders for Multi-Omics Data

Implements encoders that process individual omics modalities (RNA-seq, proteomics,
ATAC-seq, etc.) with optional graph structure integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, NamedTuple
from .gnn_layers import MultiLayerGNN


class EncoderOutput(NamedTuple):
    """Output from modality encoder"""
    z_shared: torch.Tensor      # Shared latent representation
    z_specific: torch.Tensor    # Modality-specific latent representation
    mu: torch.Tensor            # Mean for VAE
    logvar: torch.Tensor        # Log variance for VAE
    hidden: Optional[torch.Tensor] = None  # Intermediate hidden states


class ModalityEncoder(nn.Module):
    """
    Encoder for a single omics modality.

    Supports both MLP-based and GNN-based encoding depending on whether
    feature graphs are provided.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim_shared: int,
        latent_dim_specific: int,
        use_gnn: bool = False,
        gnn_config: Optional[Dict] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            latent_dim_shared: Dimension of shared latent space
            latent_dim_specific: Dimension of modality-specific latent space
            use_gnn: Whether to use GNN layers
            gnn_config: GNN configuration (if use_gnn=True)
            dropout: Dropout rate
            activation: Activation function name
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim_shared = latent_dim_shared
        self.latent_dim_specific = latent_dim_specific
        self.use_gnn = use_gnn

        # Activation function
        self.activation = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
        }[activation.lower()]

        if use_gnn and gnn_config is not None:
            # GNN-based encoder for graph-structured features
            self.feature_gnn = MultiLayerGNN(
                in_features=input_dim,
                hidden_features=hidden_dims[0],
                out_features=hidden_dims[-1],
                num_layers=gnn_config.get("num_layers", 3),
                gnn_type=gnn_config.get("type", "gcn"),
                dropout=dropout,
            )
            final_dim = hidden_dims[-1]
        else:
            # MLP-based encoder
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    self.activation,
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim

            self.mlp = nn.Sequential(*layers)
            final_dim = hidden_dims[-1]

        # Latent projection heads
        # Shared latent (for cross-modal alignment)
        self.fc_shared_mu = nn.Linear(final_dim, latent_dim_shared)
        self.fc_shared_logvar = nn.Linear(final_dim, latent_dim_shared)

        # Modality-specific latent
        self.fc_specific_mu = nn.Linear(final_dim, latent_dim_specific)
        self.fc_specific_logvar = nn.Linear(final_dim, latent_dim_specific)

    def forward(
        self,
        x: torch.Tensor,
        feature_graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> EncoderOutput:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]
            feature_graph: Optional feature graph (sparse tensor) [input_dim, input_dim]
            mask: Optional mask for missing values [batch_size, input_dim]

        Returns:
            EncoderOutput with latent representations
        """
        # Handle missing values
        if mask is not None:
            x = x * mask

        # Encode
        if self.use_gnn and feature_graph is not None:
            # GNN encoding: aggregate over feature graph
            # Note: This requires transposing since GNN operates on nodes (features)
            # x: [batch, features] -> treat each sample separately
            # For simplicity, we apply GNN to feature dimension
            h = self.feature_gnn(x.t(), feature_graph).t()  # [batch, hidden_dim]
        else:
            h = self.mlp(x)

        # Project to latent spaces
        z_shared_mu = self.fc_shared_mu(h)
        z_shared_logvar = self.fc_shared_logvar(h)

        z_specific_mu = self.fc_specific_mu(h)
        z_specific_logvar = self.fc_specific_logvar(h)

        # Sample latent variables (reparameterization trick)
        z_shared = self._reparameterize(z_shared_mu, z_shared_logvar)
        z_specific = self._reparameterize(z_specific_mu, z_specific_logvar)

        # Combine mu and logvar for loss computation
        mu = torch.cat([z_shared_mu, z_specific_mu], dim=1)
        logvar = torch.cat([z_shared_logvar, z_specific_logvar], dim=1)

        return EncoderOutput(
            z_shared=z_shared,
            z_specific=z_specific,
            mu=mu,
            logvar=logvar,
            hidden=h
        )

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class MultiOmicsEncoder(nn.Module):
    """
    Multi-omics encoder that processes all modalities and fuses them.

    Implements the hierarchical encoding strategy:
    1. Per-modality encoding (with feature-level GNN)
    2. Cross-modal attention fusion
    3. Sample-level GNN (optional)
    """

    def __init__(
        self,
        modality_configs: Dict[str, Dict],
        latent_dim_shared: int,
        use_cross_attention: bool = True,
        use_sample_gnn: bool = True,
        sample_gnn_config: Optional[Dict] = None,
    ):
        """
        Args:
            modality_configs: Dict mapping modality name to encoder config
            latent_dim_shared: Shared latent dimension across modalities
            use_cross_attention: Whether to use cross-modal attention
            use_sample_gnn: Whether to use sample-level GNN
            sample_gnn_config: Configuration for sample GNN
        """
        super().__init__()

        self.modality_names = list(modality_configs.keys())
        self.latent_dim_shared = latent_dim_shared
        self.use_cross_attention = use_cross_attention
        self.use_sample_gnn = use_sample_gnn

        # Create per-modality encoders
        self.modality_encoders = nn.ModuleDict()
        for modality, config in modality_configs.items():
            self.modality_encoders[modality] = ModalityEncoder(
                input_dim=config["input_dim"],
                hidden_dims=config.get("hidden_dims", [512, 256, 128]),
                latent_dim_shared=latent_dim_shared,
                latent_dim_specific=config.get("latent_dim_specific", 16),
                use_gnn=config.get("use_gnn", False),
                gnn_config=config.get("gnn_config"),
                dropout=config.get("dropout", 0.1),
            )

        # Cross-modal attention (optional)
        if use_cross_attention:
            from .cross_attention import CrossModalAttention
            self.cross_attention = CrossModalAttention(
                latent_dim=latent_dim_shared,
                num_heads=4,
                dropout=0.1
            )

        # Sample-level GNN (optional)
        if use_sample_gnn and sample_gnn_config is not None:
            total_latent_dim = latent_dim_shared * len(self.modality_names)
            self.sample_gnn = MultiLayerGNN(
                in_features=total_latent_dim,
                hidden_features=sample_gnn_config.get("hidden_dim", 128),
                out_features=latent_dim_shared,
                num_layers=sample_gnn_config.get("num_layers", 2),
                gnn_type=sample_gnn_config.get("type", "graphsage"),
                dropout=0.1,
            )

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        feature_graphs: Optional[Dict[str, torch.Tensor]] = None,
        sample_graph: Optional[torch.Tensor] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, EncoderOutput]:
        """
        Forward pass through all modality encoders.

        Args:
            modality_inputs: Dict mapping modality name to input tensor
            feature_graphs: Optional dict of feature graphs per modality
            sample_graph: Optional sample similarity graph
            masks: Optional masks for missing values per modality

        Returns:
            Dict mapping modality name to EncoderOutput
        """
        outputs = {}

        # Step 1: Encode each modality
        for modality in self.modality_names:
            if modality not in modality_inputs:
                continue

            x = modality_inputs[modality]
            fg = feature_graphs.get(modality) if feature_graphs else None
            mask = masks.get(modality) if masks else None

            outputs[modality] = self.modality_encoders[modality](x, fg, mask)

        # Step 2: Cross-modal attention fusion (optional)
        if self.use_cross_attention and len(outputs) > 1:
            # Collect shared latents
            z_shared_list = [out.z_shared for out in outputs.values()]

            # Apply cross-attention
            fused_shared = self.cross_attention(z_shared_list)

            # Update shared latents with fused representation
            for i, modality in enumerate(outputs.keys()):
                outputs[modality] = outputs[modality]._replace(
                    z_shared=fused_shared[i]
                )

        # Step 3: Sample-level GNN aggregation (optional)
        if self.use_sample_gnn and sample_graph is not None:
            # Concatenate all shared latents
            all_shared = torch.cat([out.z_shared for out in outputs.values()], dim=1)

            # Apply sample GNN
            fused_sample = self.sample_gnn(all_shared, sample_graph)

            # Update all shared latents with sample-level aggregation
            for modality in outputs.keys():
                outputs[modality] = outputs[modality]._replace(
                    z_shared=fused_sample
                )

        return outputs
