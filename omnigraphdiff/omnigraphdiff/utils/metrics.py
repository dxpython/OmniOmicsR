"""Evaluation metrics"""
import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compute_c_index(risk_scores, survival_time, event):
    """Compute concordance index for survival prediction."""
    return concordance_index(survival_time, -risk_scores, event)


def compute_ari(labels_true, labels_pred):
    """Compute Adjusted Rand Index."""
    return adjusted_rand_score(labels_true, labels_pred)


def compute_nmi(labels_true, labels_pred):
    """Compute Normalized Mutual Information."""
    return normalized_mutual_info_score(labels_true, labels_pred)
