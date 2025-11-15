"""
Clinical Prediction Heads

Implements Cox proportional hazards model for survival analysis
and classification heads for categorical outcomes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CoxHead(nn.Module):
    """
    Cox proportional hazards model for survival prediction.

    Predicts a risk score from latent representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input latent features
            hidden_dims: Optional hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_dims is not None and len(hidden_dims) > 0:
            # MLP before final risk prediction
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim

            self.mlp = nn.Sequential(*layers)
            final_dim = hidden_dims[-1]
        else:
            self.mlp = nn.Identity()
            final_dim = input_dim

        # Output: single risk score per sample
        self.risk_head = nn.Linear(final_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: Latent features [batch_size, input_dim]

        Returns:
            Risk scores [batch_size, 1]
        """
        h = self.mlp(z)
        risk = self.risk_head(h)  # [batch_size, 1]
        return risk


class ClassificationHead(nn.Module):
    """
    Classification head for categorical outcomes (e.g., cancer subtypes).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input latent features
            num_classes: Number of output classes
            hidden_dims: Optional hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_dims is not None and len(hidden_dims) > 0:
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim

            self.mlp = nn.Sequential(*layers)
            final_dim = hidden_dims[-1]
        else:
            self.mlp = nn.Identity()
            final_dim = input_dim

        # Output: logits for each class
        self.classifier = nn.Linear(final_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: Latent features [batch_size, input_dim]

        Returns:
            Class logits [batch_size, num_classes]
        """
        h = self.mlp(z)
        logits = self.classifier(h)
        return logits


class ClinicalHead(nn.Module):
    """
    Combined clinical prediction head supporting both survival and classification.
    """

    def __init__(
        self,
        input_dim: int,
        use_survival: bool = True,
        use_classification: bool = False,
        num_classes: Optional[int] = None,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input latent features
            use_survival: Whether to include Cox survival head
            use_classification: Whether to include classification head
            num_classes: Number of classes (if use_classification=True)
            hidden_dims: Shared hidden layers
            dropout: Dropout rate
        """
        super().__init__()

        self.use_survival = use_survival
        self.use_classification = use_classification

        # Shared feature extractor (optional)
        if hidden_dims is not None and len(hidden_dims) > 0:
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim

            self.shared_mlp = nn.Sequential(*layers)
            final_dim = hidden_dims[-1]
        else:
            self.shared_mlp = nn.Identity()
            final_dim = input_dim

        # Task-specific heads
        if use_survival:
            self.survival_head = nn.Linear(final_dim, 1)

        if use_classification:
            assert num_classes is not None, "num_classes must be specified for classification"
            self.classification_head = nn.Linear(final_dim, num_classes)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            z: Latent features [batch_size, input_dim]

        Returns:
            Dict with predictions:
                - "risk" if use_survival=True
                - "class_logits" if use_classification=True
        """
        h = self.shared_mlp(z)

        outputs = {}

        if self.use_survival:
            outputs["risk"] = self.survival_head(h)

        if self.use_classification:
            outputs["class_logits"] = self.classification_head(h)

        return outputs
