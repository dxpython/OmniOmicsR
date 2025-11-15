"""
OmniGraphDiff - Top-Level Model

Complete hierarchical graph-driven generative model combining:
- Multi-omics VAE
- Clinical prediction heads
- Graph-based regularization
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .vae import MultiOmicsVAE
from .clinical_head import ClinicalHead


class OmniGraphDiffModel(nn.Module):
    """
    Complete OmniGraphDiff model integrating all components.
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Complete model configuration
        """
        super().__init__()

        self.config = config

        # Core VAE
        self.vae = MultiOmicsVAE(config)

        # Clinical prediction head (optional)
        if config.get("use_clinical_head", True):
            self.clinical_head = ClinicalHead(
                input_dim=config["latent_dim_shared"],
                use_survival=config.get("use_survival", True),
                use_classification=config.get("use_classification", False),
                num_classes=config.get("num_classes"),
                hidden_dims=config.get("clinical_hidden_dims", [64, 32]),
                dropout=0.1,
            )
        else:
            self.clinical_head = None

    def forward(
        self,
        batch: Dict,
        return_latents: bool = False
    ) -> Dict:
        """
        Full forward pass.

        Args:
            batch: Dict containing:
                - "modalities": Dict of input tensors
                - "feature_graphs": Optional feature graphs
                - "sample_graph": Optional sample graph
                - "masks": Optional missing value masks
                - "clinical_data": Optional clinical data
            return_latents: Whether to return latent embeddings

        Returns:
            Dict with all model outputs
        """
        # VAE forward pass
        vae_outputs = self.vae(
            modality_inputs=batch["modalities"],
            feature_graphs=batch.get("feature_graphs"),
            sample_graph=batch.get("sample_graph"),
            masks=batch.get("masks"),
        )

        outputs = {
            "encoder_outputs": vae_outputs["encoder_outputs"],
            "reconstructions": vae_outputs["reconstructions"],
            "latents": vae_outputs["latents"],
        }

        # Clinical predictions (if enabled)
        if self.clinical_head is not None and "clinical_data" in batch:
            z_shared = vae_outputs["latents"]["shared"]
            clinical_preds = self.clinical_head(z_shared)
            outputs["clinical_predictions"] = clinical_preds

        return outputs

    def encode(self, batch: Dict) -> Dict:
        """Encode inputs to latent space."""
        vae_outputs = self.vae(
            modality_inputs=batch["modalities"],
            feature_graphs=batch.get("feature_graphs"),
            sample_graph=batch.get("sample_graph"),
            masks=batch.get("masks"),
        )
        return vae_outputs["latents"]

    def decode(
        self,
        latents: Dict,
        feature_graphs: Optional[Dict] = None
    ) -> Dict:
        """Decode latent codes to reconstructions."""
        # Prepare encoder outputs format
        encoder_outputs = {}
        for modality in latents["specific"].keys():
            from .encoders import EncoderOutput
            encoder_outputs[modality] = EncoderOutput(
                z_shared=latents["shared"],
                z_specific=latents["specific"][modality],
                mu=latents["mu"][modality],
                logvar=latents["logvar"][modality],
            )

        return self.vae.decode(encoder_outputs, feature_graphs)
