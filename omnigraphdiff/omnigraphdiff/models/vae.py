"""
Multi-Omics Variational Autoencoder

Complete implementation of the hierarchical Graph-VAE for multi-omics integration.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .encoders import MultiOmicsEncoder
from .decoders import MultiOmicsDecoder


class MultiOmicsVAE(nn.Module):
    """
    Hierarchical Graph-VAE for multi-omics integration.

    Implements the complete VAE architecture with:
    - Per-modality encoders with graph structure
    - Cross-modal attention fusion
    - Sample-level GNN aggregation
    - Shared + modality-specific latent variables
    - Per-modality decoders
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary containing:
                - modalities: Dict mapping modality names to configs
                - latent_dim_shared: Shared latent dimension
                - use_cross_attention: Whether to use cross-modal attention
                - use_sample_gnn: Whether to use sample-level GNN
                - sample_gnn_config: Sample GNN configuration
        """
        super().__init__()

        self.modality_names = list(config["modalities"].keys())
        self.latent_dim_shared = config["latent_dim_shared"]

        # Build encoder configs
        encoder_configs = {}
        for modality, mod_config in config["modalities"].items():
            encoder_configs[modality] = {
                "input_dim": mod_config["input_dim"],
                "hidden_dims": mod_config.get("encoder_hidden_dims", [512, 256, 128]),
                "latent_dim_specific": mod_config.get("latent_dim_specific", 16),
                "use_gnn": mod_config.get("use_feature_gnn", False),
                "gnn_config": mod_config.get("feature_gnn_config"),
                "dropout": mod_config.get("dropout", 0.1),
            }

        # Encoder
        self.encoder = MultiOmicsEncoder(
            modality_configs=encoder_configs,
            latent_dim_shared=self.latent_dim_shared,
            use_cross_attention=config.get("use_cross_attention", True),
            use_sample_gnn=config.get("use_sample_gnn", True),
            sample_gnn_config=config.get("sample_gnn_config"),
        )

        # Build decoder configs
        decoder_configs = {}
        for modality, mod_config in config["modalities"].items():
            decoder_configs[modality] = {
                "output_dim": mod_config["input_dim"],  # Reconstruct original dims
                "hidden_dims": mod_config.get("decoder_hidden_dims", [128, 256, 512]),
                "latent_dim_specific": mod_config.get("latent_dim_specific", 16),
                "output_dist": mod_config.get("output_dist", "gaussian"),
                "use_gnn": mod_config.get("use_feature_gnn", False),
                "gnn_config": mod_config.get("feature_gnn_config"),
                "dropout": mod_config.get("dropout", 0.1),
            }

        # Decoder
        self.decoder = MultiOmicsDecoder(
            modality_configs=decoder_configs,
            latent_dim_shared=self.latent_dim_shared,
        )

    def encode(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        feature_graphs: Optional[Dict[str, torch.Tensor]] = None,
        sample_graph: Optional[torch.Tensor] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict:
        """Encode all modalities to latent space."""
        return self.encoder(modality_inputs, feature_graphs, sample_graph, masks)

    def decode(
        self,
        encoder_outputs: Dict,
        feature_graphs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict:
        """Decode latent codes to reconstructions."""
        # Prepare latent codes for decoder
        latent_codes = {}
        for modality, output in encoder_outputs.items():
            latent_codes[modality] = {
                "z_shared": output.z_shared,
                "z_specific": output.z_specific,
            }

        return self.decoder(latent_codes, feature_graphs)

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        feature_graphs: Optional[Dict[str, torch.Tensor]] = None,
        sample_graph: Optional[torch.Tensor] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict:
        """
        Full forward pass.

        Returns:
            Dict containing:
                - "encoder_outputs": Encoder outputs per modality
                - "reconstructions": Reconstructed data per modality
                - "latents": Latent representations (for downstream tasks)
        """
        # Encode
        encoder_outputs = self.encode(
            modality_inputs, feature_graphs, sample_graph, masks
        )

        # Decode
        reconstructions = self.decode(encoder_outputs, feature_graphs)

        # Collect latent representations (for downstream use)
        latents = {
            "shared": torch.stack([out.z_shared for out in encoder_outputs.values()]).mean(0),
            "specific": {mod: out.z_specific for mod, out in encoder_outputs.items()},
            "mu": {mod: out.mu for mod, out in encoder_outputs.items()},
            "logvar": {mod: out.logvar for mod, out in encoder_outputs.items()},
        }

        return {
            "encoder_outputs": encoder_outputs,
            "reconstructions": reconstructions,
            "latents": latents,
        }
