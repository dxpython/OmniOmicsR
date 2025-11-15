"""
Complete Loss Function Implementation

Implements all 5 loss components from MODEL_DESIGN.md:
1. Reconstruction loss (NB, MSE, BCE)
2. KL divergence
3. Graph regularization (Laplacian smoothness)
4. Clinical prediction loss (Cox + classification)
5. Contrastive loss (InfoNCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


# ==================== Component 1: Reconstruction Loss ====================

def negative_binomial_loss(
    x_true: torch.Tensor,
    mean: torch.Tensor,
    dispersion: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Negative binomial loss for count data (RNA-seq).

    NB(x | μ, θ) where θ is dispersion parameter.
    """
    # Clip to avoid numerical issues
    mean = torch.clamp(mean, min=eps)
    dispersion = torch.clamp(dispersion, min=eps)

    # NB log-likelihood
    t1 = torch.lgamma(dispersion + eps) + torch.lgamma(x_true + 1.0)
    t2 = torch.lgamma(x_true + dispersion + eps)
    t3 = (dispersion + x_true) * torch.log(1.0 + (mean / (dispersion + eps)))
    t4 = x_true * torch.log(mean / (dispersion + eps) + eps)

    loss = t1 - t2 + t3 - t4
    return loss.sum(dim=-1).mean()


def gaussian_loss(
    x_true: torch.Tensor,
    mean: torch.Tensor,
    logvar: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Gaussian reconstruction loss (MSE or negative log-likelihood).
    """
    if logvar is None:
        # Simple MSE
        return F.mse_loss(mean, x_true, reduction="mean")
    else:
        # Full Gaussian NLL
        var = torch.exp(logvar)
        loss = 0.5 * (logvar + ((x_true - mean) ** 2) / var + math.log(2 * math.pi))
        return loss.sum(dim=-1).mean()


def bernoulli_loss(
    x_true: torch.Tensor,
    logits: torch.Tensor
) -> torch.Tensor:
    """
    Bernoulli loss for binary data (ATAC-seq peaks).
    """
    return F.binary_cross_entropy_with_logits(logits, x_true, reduction="mean")


def reconstruction_loss(
    targets: Dict[str, torch.Tensor],
    reconstructions: Dict[str, Dict[str, torch.Tensor]],
    modality_configs: Dict[str, Dict]
) -> Dict[str, torch.Tensor]:
    """
    Compute per-modality reconstruction loss.

    Args:
        targets: Dict mapping modality to target data
        reconstructions: Dict mapping modality to reconstruction params
        modality_configs: Dict with output_dist specification per modality

    Returns:
        Dict mapping modality to loss value
    """
    losses = {}

    for modality in targets.keys():
        if modality not in reconstructions:
            continue

        x_true = targets[modality]
        recon_params = reconstructions[modality]
        output_dist = modality_configs[modality].get("output_dist", "gaussian")

        if output_dist == "nb":
            loss = negative_binomial_loss(
                x_true,
                recon_params["mean"],
                recon_params["dispersion"]
            )
        elif output_dist == "gaussian":
            loss = gaussian_loss(
                x_true,
                recon_params["mean"],
                recon_params.get("logvar")
            )
        elif output_dist == "bernoulli":
            loss = bernoulli_loss(x_true, recon_params["logits"])
        else:
            raise ValueError(f"Unknown output_dist: {output_dist}")

        losses[modality] = loss

    return losses


# ==================== Component 2: KL Divergence ====================

def kl_divergence_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_mu: Optional[torch.Tensor] = None,
    prior_logvar: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    KL divergence between q(z|x) and p(z).

    Assuming Gaussian distributions:
    KL(q || p) = -0.5 * sum(1 + logvar_q - logvar_p - (mu_q - mu_p)^2/var_p - var_q/var_p)

    If prior is N(0, I), simplifies to:
    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    if prior_mu is None:
        # Standard normal prior N(0, I)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    else:
        # General Gaussian prior
        if prior_logvar is None:
            prior_logvar = torch.zeros_like(prior_mu)

        var_ratio = torch.exp(logvar - prior_logvar)
        mu_diff_sq = (mu - prior_mu).pow(2) * torch.exp(-prior_logvar)

        kl = -0.5 * torch.sum(
            1 + logvar - prior_logvar - var_ratio - mu_diff_sq,
            dim=-1
        )

    return kl.mean()


# ==================== Component 3: Graph Regularization ====================

def graph_regularization_loss(
    embeddings: torch.Tensor,
    laplacian: torch.Tensor
) -> torch.Tensor:
    """
    Graph Laplacian smoothness regularization.

    L_graph = tr(Z^T L Z)

    Encourages embeddings to be smooth over graph structure.

    Args:
        embeddings: Node embeddings [N, D]
        laplacian: Graph Laplacian (sparse tensor) [N, N]

    Returns:
        Scalar loss
    """
    # Compute Z^T L Z
    if laplacian.layout == torch.sparse_coo:
        # Sparse-dense multiplication
        Lz = torch.sparse.mm(laplacian, embeddings)  # [N, D]
    else:
        Lz = torch.mm(laplacian, embeddings)

    # tr(Z^T L Z) = sum(Z * LZ)
    loss = torch.sum(embeddings * Lz)

    return loss / embeddings.size(0)  # Normalize by number of nodes


# ==================== Component 4: Clinical Loss ====================

def cox_partial_likelihood_loss(
    risk_scores: torch.Tensor,
    survival_time: torch.Tensor,
    event: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Cox proportional hazards partial likelihood loss.

    L = -sum_{i: δ_i=1} [ β^T z_i - log sum_{j: T_j >= T_i} exp(β^T z_j) ]

    Args:
        risk_scores: Predicted risk scores [batch_size, 1]
        survival_time: Observed survival times [batch_size]
        event: Event indicator (1=event, 0=censored) [batch_size]

    Returns:
        Scalar loss
    """
    # Sort by survival time (descending)
    sorted_idx = torch.argsort(survival_time, descending=True)
    risk_sorted = risk_scores[sorted_idx].squeeze()
    event_sorted = event[sorted_idx]

    # Compute cumulative sum of exp(risk) (risk set)
    exp_risk = torch.exp(risk_sorted)
    cumsum_exp_risk = torch.cumsum(exp_risk, dim=0)

    # Log partial likelihood for events only
    log_likelihood = risk_sorted - torch.log(cumsum_exp_risk + eps)
    log_likelihood = log_likelihood * event_sorted  # Mask non-events

    # Negative log-likelihood
    loss = -torch.sum(log_likelihood) / (torch.sum(event_sorted) + eps)

    return loss


def clinical_loss(
    predictions: Dict[str, torch.Tensor],
    clinical_data: Dict[str, torch.Tensor],
    use_survival: bool = True,
    use_classification: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Combined clinical prediction loss.

    Args:
        predictions: Dict with "risk" and/or "class_logits"
        clinical_data: Dict with "survival_time", "event", "class_labels"
        use_survival: Whether to compute Cox loss
        use_classification: Whether to compute classification loss

    Returns:
        Dict with loss components
    """
    losses = {}

    if use_survival and "risk" in predictions:
        loss_cox = cox_partial_likelihood_loss(
            predictions["risk"],
            clinical_data["survival_time"],
            clinical_data["event"]
        )
        losses["cox"] = loss_cox

    if use_classification and "class_logits" in predictions:
        loss_ce = F.cross_entropy(
            predictions["class_logits"],
            clinical_data["class_labels"]
        )
        losses["classification"] = loss_ce

    return losses


# ==================== Component 5: Contrastive Loss ====================

def info_nce_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE contrastive loss.

    Encourages alignment between anchor and positive while pushing away negatives.

    Args:
        z_anchor: Anchor embeddings [batch_size, dim]
        z_positive: Positive embeddings [batch_size, dim] (from same sample)
        temperature: Temperature parameter

    Returns:
        Scalar loss
    """
    batch_size = z_anchor.size(0)

    # Normalize embeddings
    z_anchor = F.normalize(z_anchor, dim=1)
    z_positive = F.normalize(z_positive, dim=1)

    # Compute similarity matrix [batch, batch]
    sim_matrix = torch.mm(z_anchor, z_positive.t()) / temperature

    # Labels: diagonal elements are positives
    labels = torch.arange(batch_size, device=z_anchor.device)

    # Cross-entropy loss (each row should predict its own index)
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def contrastive_loss(
    z_shared: torch.Tensor,
    z_specific_dict: Dict[str, torch.Tensor],
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Multi-modal contrastive loss.

    Aligns shared latent with each modality-specific latent.

    Args:
        z_shared: Shared latent representation [batch_size, dim_shared]
        z_specific_dict: Dict mapping modality to specific latent [batch, dim_spec]
        temperature: Temperature parameter

    Returns:
        Average InfoNCE loss across modalities
    """
    losses = []

    for modality, z_spec in z_specific_dict.items():
        loss = info_nce_loss(z_shared, z_spec, temperature)
        losses.append(loss)

    return torch.stack(losses).mean()


# ==================== Total Loss ====================

def compute_total_loss(
    model_outputs: Dict,
    batch: Dict,
    config: Dict,
    feature_graphs: Optional[Dict] = None,
    sample_graph: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute total multi-objective loss.

    L_total = L_recon + λ1*L_KL + λ2*L_graph + λ3*L_clinical + λ4*L_contrastive

    Args:
        model_outputs: Outputs from model forward pass
        batch: Batch data (targets, clinical data, etc.)
        config: Configuration with loss weights
        feature_graphs: Optional feature graphs for regularization
        sample_graph: Optional sample graph for regularization

    Returns:
        total_loss: Scalar total loss
        loss_dict: Dict with individual loss components
    """
    loss_dict = {}

    # Extract components
    encoder_outputs = model_outputs["encoder_outputs"]
    reconstructions = model_outputs["reconstructions"]
    latents = model_outputs["latents"]

    # ===== 1. Reconstruction Loss =====
    recon_losses = reconstruction_loss(
        batch["targets"],
        reconstructions,
        config["modalities"]
    )
    loss_recon = sum(recon_losses.values())
    loss_dict["recon"] = loss_recon
    for mod, loss in recon_losses.items():
        loss_dict[f"recon_{mod}"] = loss

    # ===== 2. KL Divergence =====
    kl_losses = []
    for modality, output in encoder_outputs.items():
        kl = kl_divergence_loss(output.mu, output.logvar)
        kl_losses.append(kl)
    loss_kl = torch.stack(kl_losses).mean()
    loss_dict["kl"] = loss_kl

    # ===== 3. Graph Regularization =====
    loss_graph = torch.tensor(0.0, device=loss_recon.device)

    # Feature-level graph regularization
    if feature_graphs is not None and config.get("use_feature_graph_reg", True):
        for modality, laplacian in feature_graphs.items():
            if modality in encoder_outputs:
                # Use modality-specific latent for feature graph reg
                z_spec = encoder_outputs[modality].z_specific
                loss_graph += graph_regularization_loss(z_spec, laplacian)

    # Sample-level graph regularization
    if sample_graph is not None and config.get("use_sample_graph_reg", True):
        z_shared = latents["shared"]
        loss_graph += graph_regularization_loss(z_shared, sample_graph)

    loss_dict["graph"] = loss_graph

    # ===== 4. Clinical Loss =====
    loss_clinical = torch.tensor(0.0, device=loss_recon.device)

    if "clinical_predictions" in model_outputs and "clinical_data" in batch:
        clinical_losses = clinical_loss(
            model_outputs["clinical_predictions"],
            batch["clinical_data"],
            use_survival=config.get("use_survival", True),
            use_classification=config.get("use_classification", False)
        )
        loss_clinical = sum(clinical_losses.values())
        loss_dict.update({f"clinical_{k}": v for k, v in clinical_losses.items()})

    loss_dict["clinical"] = loss_clinical

    # ===== 5. Contrastive Loss =====
    loss_contrastive = torch.tensor(0.0, device=loss_recon.device)

    if config.get("use_contrastive", True) and len(latents["specific"]) > 1:
        loss_contrastive = contrastive_loss(
            latents["shared"],
            latents["specific"],
            temperature=config.get("contrastive_temperature", 0.07)
        )
    loss_dict["contrastive"] = loss_contrastive

    # ===== Total Loss =====
    weights = config.get("loss_weights", {})
    lambda_kl = weights.get("kl", 0.5)
    lambda_graph = weights.get("graph", 0.1)
    lambda_clinical = weights.get("clinical", 1.0)
    lambda_contrastive = weights.get("contrastive", 0.5)

    total_loss = (
        loss_recon +
        lambda_kl * loss_kl +
        lambda_graph * loss_graph +
        lambda_clinical * loss_clinical +
        lambda_contrastive * loss_contrastive
    )

    loss_dict["total"] = total_loss

    return total_loss, loss_dict
