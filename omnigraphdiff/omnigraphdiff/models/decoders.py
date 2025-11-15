"""
Modality-Specific Decoders for Multi-Omics Reconstruction

Implements decoders that reconstruct omics data from latent representations,
with support for different data types (counts, continuous, binary).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal
from .gnn_layers import MultiLayerGNN


class ModalityDecoder(nn.Module):
    """
    Decoder for a single omics modality.

    Supports different output distributions:
    - 'nb' (negative binomial) for RNA-seq counts
    - 'gaussian' for continuous data (proteomics, metabolomics)
    - 'bernoulli' for binary data (ATAC-seq peaks)
    """

    def __init__(
        self,
        latent_dim: int,          # Total latent dim (shared + specific)
        hidden_dims: list[int],
        output_dim: int,
        output_dist: Literal["nb", "gaussian", "bernoulli"] = "gaussian",
        use_gnn: bool = False,
        gnn_config: Optional[Dict] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            latent_dim: Combined latent dimension (shared + specific)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output feature dimension
            output_dist: Output distribution type
            use_gnn: Whether to use GNN for reconstruction
            gnn_config: GNN configuration
            dropout: Dropout rate
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.output_dist = output_dist
        self.use_gnn = use_gnn

        if use_gnn and gnn_config is not None:
            # GNN-based decoder
            self.feature_gnn = MultiLayerGNN(
                in_features=latent_dim,
                hidden_features=hidden_dims[0],
                out_features=hidden_dims[-1],
                num_layers=gnn_config.get("num_layers", 2),
                gnn_type=gnn_config.get("type", "gcn"),
                dropout=dropout,
            )
            final_dim = hidden_dims[-1]
        else:
            # MLP-based decoder
            layers = []
            prev_dim = latent_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim

            self.mlp = nn.Sequential(*layers)
            final_dim = hidden_dims[-1]

        # Output heads depending on distribution
        if output_dist == "nb":
            # Negative binomial: predict mean and dispersion
            self.mean_decoder = nn.Sequential(
                nn.Linear(final_dim, output_dim),
                nn.Softplus()  # Ensure positive mean
            )
            self.dispersion_decoder = nn.Sequential(
                nn.Linear(final_dim, output_dim),
                nn.Softplus()  # Ensure positive dispersion
            )

        elif output_dist == "gaussian":
            # Gaussian: predict mean (variance can be fixed or learned)
            self.mean_decoder = nn.Linear(final_dim, output_dim)
            self.logvar_decoder = nn.Linear(final_dim, output_dim)  # Optional

        elif output_dist == "bernoulli":
            # Bernoulli: predict logits
            self.logit_decoder = nn.Linear(final_dim, output_dim)

        else:
            raise ValueError(f"Unknown output_dist: {output_dist}")

    def forward(
        self,
        z: torch.Tensor,
        feature_graph: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            z: Latent representation [batch_size, latent_dim]
            feature_graph: Optional feature graph

        Returns:
            Dict with reconstruction parameters (distribution-specific)
        """
        # Decode
        if self.use_gnn and feature_graph is not None:
            h = self.feature_gnn(z.t(), feature_graph).t()
        else:
            h = self.mlp(z)

        # Generate output parameters
        if self.output_dist == "nb":
            mean = self.mean_decoder(h)
            dispersion = self.dispersion_decoder(h)
            return {"mean": mean, "dispersion": dispersion}

        elif self.output_dist == "gaussian":
            mean = self.mean_decoder(h)
            logvar = self.logvar_decoder(h)
            return {"mean": mean, "logvar": logvar}

        elif self.output_dist == "bernoulli":
            logits = self.logit_decoder(h)
            probs = torch.sigmoid(logits)
            return {"logits": logits, "probs": probs}


class MultiOmicsDecoder(nn.Module):
    """
    Multi-omics decoder that reconstructs all modalities from latent codes.
    """

    def __init__(
        self,
        modality_configs: Dict[str, Dict],
        latent_dim_shared: int,
    ):
        """
        Args:
            modality_configs: Dict mapping modality name to decoder config
            latent_dim_shared: Shared latent dimension
        """
        super().__init__()

        self.modality_names = list(modality_configs.keys())
        self.latent_dim_shared = latent_dim_shared

        # Create per-modality decoders
        self.modality_decoders = nn.ModuleDict()
        for modality, config in modality_configs.items():
            total_latent_dim = latent_dim_shared + config.get("latent_dim_specific", 16)

            self.modality_decoders[modality] = ModalityDecoder(
                latent_dim=total_latent_dim,
                hidden_dims=config.get("hidden_dims", [128, 256, 512]),
                output_dim=config["output_dim"],
                output_dist=config.get("output_dist", "gaussian"),
                use_gnn=config.get("use_gnn", False),
                gnn_config=config.get("gnn_config"),
                dropout=config.get("dropout", 0.1),
            )

    def forward(
        self,
        latent_codes: Dict[str, Dict[str, torch.Tensor]],
        feature_graphs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Decode all modalities.

        Args:
            latent_codes: Dict mapping modality to {"z_shared": ..., "z_specific": ...}
            feature_graphs: Optional feature graphs per modality

        Returns:
            Dict mapping modality to reconstruction parameters
        """
        reconstructions = {}

        for modality in self.modality_names:
            if modality not in latent_codes:
                continue

            # Concatenate shared and specific latents
            z_shared = latent_codes[modality]["z_shared"]
            z_specific = latent_codes[modality]["z_specific"]
            z = torch.cat([z_shared, z_specific], dim=1)

            # Decode
            fg = feature_graphs.get(modality) if feature_graphs else None
            reconstructions[modality] = self.modality_decoders[modality](z, fg)

        return reconstructions
