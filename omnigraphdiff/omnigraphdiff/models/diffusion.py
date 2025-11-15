"""
Graph-Conditioned Diffusion Model for Multi-Omics

Complete implementation of denoising diffusion probabilistic model (DDPM)
with graph-based conditioning for multi-omics data generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math
import numpy as np


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time step embedding for diffusion models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [batch_size] or scalar

        Returns:
            embeddings: [batch_size, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class GraphUNet(nn.Module):
    """
    U-Net style architecture with graph convolutions for noise prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        time_dim: int = 128,
        cond_dim: Optional[int] = None,
        use_gnn: bool = True,
        gnn_type: str = "gcn",
    ):
        super().__init__()

        self.use_gnn = use_gnn

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Conditioning embedding (optional)
        if cond_dim is not None:
            self.cond_proj = nn.Linear(cond_dim, time_dim)
        else:
            self.cond_proj = None

        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            if use_gnn:
                from .gnn_layers import GCNLayer
                block = GCNLayer(
                    prev_dim + time_dim,  # Concatenate time embedding
                    hidden_dim,
                    dropout=0.1
                )
            else:
                block = nn.Sequential(
                    nn.Linear(prev_dim + time_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                )
            self.down_blocks.append(block)
            prev_dim = hidden_dim

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(prev_dim + time_dim, prev_dim),
            nn.GELU(),
            nn.Linear(prev_dim, prev_dim),
        )

        # Decoder (upsampling with skip connections)
        self.up_blocks = nn.ModuleList()
        for i, hidden_dim in enumerate(reversed(hidden_dims[:-1])):
            if use_gnn:
                from .gnn_layers import GCNLayer
                block = GCNLayer(
                    prev_dim + hidden_dims[-(i+2)] + time_dim,  # Skip + time
                    hidden_dim,
                    dropout=0.1
                )
            else:
                block = nn.Sequential(
                    nn.Linear(prev_dim + hidden_dims[-(i+2)] + time_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                )
            self.up_blocks.append(block)
            prev_dim = hidden_dim

        # Output projection
        self.output_proj = nn.Linear(prev_dim + time_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict noise.

        Args:
            x: Noisy input [batch_size, input_dim]
            t: Timestep [batch_size]
            graph: Optional graph structure (sparse tensor)
            condition: Optional conditioning [batch_size, cond_dim]

        Returns:
            Predicted noise [batch_size, input_dim]
        """
        # Time embedding
        t_emb = self.time_mlp(t)

        # Conditioning
        if condition is not None and self.cond_proj is not None:
            t_emb = t_emb + self.cond_proj(condition)

        # Encoder with skip connections
        skips = []
        h = x

        for block in self.down_blocks:
            # Concatenate time embedding
            h_time = torch.cat([h, t_emb], dim=-1)

            if self.use_gnn and graph is not None:
                h = block(h_time, graph)
            else:
                h = block(h_time)

            skips.append(h)

        # Bottleneck
        h_time = torch.cat([h, t_emb], dim=-1)
        h = self.bottleneck(h_time)

        # Decoder with skip connections
        for i, block in enumerate(self.up_blocks):
            # Add skip connection
            skip = skips[-(i+2)]  # Reverse order
            h_skip = torch.cat([h, skip, t_emb], dim=-1)

            if self.use_gnn and graph is not None:
                h = block(h_skip, graph)
            else:
                h = block(h_skip)

        # Output
        h_final = torch.cat([h, t_emb], dim=-1)
        noise_pred = self.output_proj(h_final)

        return noise_pred


class GraphDiffusionModel(nn.Module):
    """
    Complete graph-conditioned diffusion model.
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.num_timesteps = config.get("num_timesteps", 1000)

        # Create noise schedule
        self.register_buffer(
            "betas",
            self._create_noise_schedule(
                self.num_timesteps,
                schedule=config.get("noise_schedule", "linear")
            )
        )

        # Pre-compute useful quantities
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior variance
        posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                            self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        # Noise prediction network (per modality)
        self.denoisers = nn.ModuleDict()
        for modality, mod_config in config["modalities"].items():
            self.denoisers[modality] = GraphUNet(
                input_dim=mod_config["input_dim"],
                hidden_dims=mod_config.get("diffusion_hidden_dims", [256, 128, 64]),
                time_dim=config.get("time_embedding_dim", 128),
                cond_dim=mod_config.get("condition_dim"),
                use_gnn=mod_config.get("use_feature_gnn", False),
                gnn_type=mod_config.get("gnn_type", "gcn"),
            )

    def _create_noise_schedule(
        self,
        num_timesteps: int,
        schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ) -> torch.Tensor:
        """Create noise schedule."""
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: Add noise to x_start at timestep t.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t][:, None]
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t][:, None]

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self.posterior_mean_coef1[t][:, None] * x_start +
            self.posterior_mean_coef2[t][:, None] * x_t
        )
        posterior_variance = self.posterior_variance[t][:, None]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t][:, None]

        return posterior_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(
        self,
        modality: str,
        x_t: torch.Tensor,
        t: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from x_t.

        p(x_{t-1} | x_t) approximation.
        """
        # Predict noise
        noise_pred = self.denoisers[modality](x_t, t, graph, condition)

        # Predict x_0
        x_start = self.predict_start_from_noise(x_t, t, noise_pred)

        # Compute posterior mean and variance
        posterior_mean, posterior_log_variance = self.q_posterior(x_start, x_t, t)

        # Sample (no noise at t=0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float()[:, None]

        return posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        modality: str,
        batch_size: int,
        device: torch.device,
        graph: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Generate samples by reverse diffusion.

        Args:
            modality: Which modality to generate
            batch_size: Number of samples
            device: Device to use
            graph: Optional graph structure
            condition: Optional conditioning
            return_intermediates: Whether to return all intermediate steps

        Returns:
            Generated samples [batch_size, input_dim]
        """
        input_dim = self.config["modalities"][modality]["input_dim"]

        # Start from pure noise
        x = torch.randn(batch_size, input_dim, device=device)

        intermediates = [x] if return_intermediates else None

        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(modality, x, t_batch, graph, condition)

            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return x, intermediates
        return x

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        feature_graphs: Optional[Dict[str, torch.Tensor]] = None,
        conditions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass: predict noise for random timesteps.

        Args:
            modality_inputs: Dict mapping modality to data
            feature_graphs: Optional feature graphs
            conditions: Optional conditioning per modality

        Returns:
            Dict of predicted noise per modality
        """
        batch_size = list(modality_inputs.values())[0].size(0)
        device = list(modality_inputs.values())[0].device

        # Random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

        losses = {}
        noise_preds = {}

        for modality, x_start in modality_inputs.items():
            # Add noise
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start, t, noise)

            # Predict noise
            graph = feature_graphs.get(modality) if feature_graphs else None
            cond = conditions.get(modality) if conditions else None

            noise_pred = self.denoisers[modality](x_t, t, graph, cond)
            noise_preds[modality] = noise_pred

            # Simple MSE loss
            losses[modality] = F.mse_loss(noise_pred, noise)

        return {
            "noise_predictions": noise_preds,
            "losses": losses,
            "total_loss": sum(losses.values())
        }
