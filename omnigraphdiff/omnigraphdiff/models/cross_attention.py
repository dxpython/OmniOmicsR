"""
Cross-Modal Attention for Multi-Omics Fusion

Implements multi-head attention mechanism to fuse information across modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math


class CrossModalAttention(nn.Module):
    """
    Multi-head cross-attention for fusing multiple modalities.

    Each modality attends to all other modalities to exchange information.
    """

    def __init__(
        self,
        latent_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            latent_dim: Dimension of latent representations
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        # Query, Key, Value projections (per modality)
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)

        # Output projection
        self.out_proj = nn.Linear(latent_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, modality_embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply cross-modal attention.

        Args:
            modality_embeddings: List of embeddings, one per modality [batch, latent_dim]

        Returns:
            List of fused embeddings, same length as input
        """
        num_modalities = len(modality_embeddings)
        batch_size = modality_embeddings[0].size(0)

        # Stack modalities: [num_modalities, batch, latent_dim]
        stacked = torch.stack(modality_embeddings, dim=0)

        # Project to Q, K, V
        Q = self.q_proj(stacked)  # [num_modalities, batch, latent_dim]
        K = self.k_proj(stacked)
        V = self.v_proj(stacked)

        # Reshape for multi-head attention
        # [num_modalities, batch, num_heads, head_dim]
        Q = Q.view(num_modalities, batch_size, self.num_heads, self.head_dim)
        K = K.view(num_modalities, batch_size, self.num_heads, self.head_dim)
        V = V.view(num_modalities, batch_size, self.num_heads, self.head_dim)

        # Transpose to [batch, num_heads, num_modalities, head_dim]
        Q = Q.permute(1, 2, 0, 3)
        K = K.permute(1, 2, 0, 3)
        V = V.permute(1, 2, 0, 3)

        # Compute attention scores
        # [batch, num_heads, num_modalities, num_modalities]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Softmax over modalities (dim=-1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [batch, num_heads, num_modalities, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back to [batch, num_modalities, latent_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, num_modalities, self.latent_dim)

        # Output projection
        output = self.out_proj(attn_output)  # [batch, num_modalities, latent_dim]

        # Split back into list of modality embeddings
        fused_embeddings = [output[:, i, :] for i in range(num_modalities)]

        return fused_embeddings
