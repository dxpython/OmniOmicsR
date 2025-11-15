"""
Graph Neural Network Layers

Implements GCN, GAT, GraphSAGE, and other GNN architectures
with integration to C++ backend for sparse operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import math

try:
    import omnigraph_cpp
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer (Kipf & Welling, 2017)

    H' = σ(D̃^(-1/2) à D̃^(-1/2) H W)

    where à = A + I (adjacency with self-loops)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[Callable] = F.relu,
        dropout: float = 0.0,
        use_cpp_backend: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_cpp_backend = use_cpp_backend and BACKEND_AVAILABLE

        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix (sparse tensor) [N, N]

        Returns:
            Updated node features [N, out_features]
        """
        # Apply dropout to input features
        if self.dropout is not None:
            x = self.dropout(x)

        # Transform features: H @ W
        support = torch.mm(x, self.weight)

        # Aggregate: Ã @ (H @ W)
        if self.use_cpp_backend and adj.layout == torch.sparse_coo:
            # Use C++ backend for sparse mm
            from ..utils import torch_sparse_to_cpp
            adj_cpp = torch_sparse_to_cpp(adj)
            output = omnigraph_cpp.sparse_mm(adj_cpp, support)
        else:
            # Fallback to PyTorch sparse mm
            output = torch.sparse.mm(adj, support)

        # Add bias
        if self.bias is not None:
            output = output + self.bias

        # Activation
        if self.activation is not None:
            output = self.activation(output)

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE Layer (Hamilton et al., 2017)

    H' = σ(W · [h_i || AGG({h_j : j ∈ N(i)})])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator: str = 'mean',  # 'mean', 'max', 'lstm', 'pool'
        bias: bool = True,
        activation: Optional[Callable] = F.relu,
        dropout: float = 0.0,
        normalize: bool = True,
        use_cpp_backend: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        self.normalize = normalize
        self.use_cpp_backend = use_cpp_backend and BACKEND_AVAILABLE

        # Weight for concatenated [self || neighbor]
        self.weight = nn.Parameter(torch.FloatTensor(in_features * 2, out_features))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Aggregator-specific parameters
        if aggregator == 'pool':
            self.pool_mlp = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.ReLU()
            )
        elif aggregator == 'lstm':
            self.lstm = nn.LSTM(in_features, in_features, batch_first=True)

        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def aggregate_neighbors(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Aggregate neighbor features."""

        if self.aggregator == 'mean':
            # Mean aggregation
            if self.use_cpp_backend:
                from ..utils import torch_sparse_to_cpp
                adj_cpp = torch_sparse_to_cpp(adj)
                agg = omnigraph_cpp.MessagePassing.aggregate_mean(adj_cpp, x)
            else:
                # Normalize by degree
                degree = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(1) + 1e-6
                agg = torch.sparse.mm(adj, x) / degree

        elif self.aggregator == 'max':
            # Max aggregation (requires dense or special handling)
            if self.use_cpp_backend:
                from ..utils import torch_sparse_to_cpp
                adj_cpp = torch_sparse_to_cpp(adj)
                agg = omnigraph_cpp.MessagePassing.aggregate_max(adj_cpp, x)
            else:
                # Fallback: convert to dense (inefficient for large graphs)
                adj_dense = adj.to_dense()
                neighbor_features = adj_dense.unsqueeze(-1) * x.unsqueeze(0)
                agg, _ = neighbor_features.max(dim=1)

        elif self.aggregator == 'pool':
            # Pooling aggregation
            x_pool = self.pool_mlp(x)
            degree = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(1) + 1e-6
            agg = torch.sparse.mm(adj, x_pool) / degree

        elif self.aggregator == 'lstm':
            # LSTM aggregation (complex, requires sequence)
            # Simplified: just use mean for now
            degree = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(1) + 1e-6
            agg = torch.sparse.mm(adj, x) / degree

        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        return agg

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix (sparse) [N, N]

        Returns:
            Updated features [N, out_features]
        """
        if self.dropout is not None:
            x = self.dropout(x)

        # Aggregate neighbors
        neighbor_agg = self.aggregate_neighbors(x, adj)

        # Concatenate self + aggregated neighbors
        concat = torch.cat([x, neighbor_agg], dim=1)

        # Transform
        output = torch.mm(concat, self.weight)

        if self.bias is not None:
            output = output + self.bias

        # L2 normalization
        if self.normalize:
            output = F.normalize(output, p=2, dim=1)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, aggregator={self.aggregator})"


class GATLayer(nn.Module):
    """
    Graph Attention Network Layer (Veličković et al., 2018)

    Uses attention mechanism to weight neighbor contributions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        concat: bool = True,
        bias: bool = True,
        activation: Optional[Callable] = F.elu,
        dropout: float = 0.0,
        alpha: float = 0.2  # LeakyReLU negative slope
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Weight matrices for each head
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))

        # Attention parameters
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_features, 1))

        if bias and concat:
            self.bias = nn.Parameter(torch.FloatTensor(num_heads * out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-head attention.

        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix (sparse) [N, N]

        Returns:
            Updated features [N, out_features * num_heads] if concat
                         or [N, out_features] if averaged
        """
        N = x.size(0)

        if self.dropout is not None:
            x = self.dropout(x)

        # Transform features for each head: [num_heads, N, out_features]
        h = torch.stack([torch.mm(x, self.W[i]) for i in range(self.num_heads)])

        # Compute attention coefficients
        # a_input = [h_i || h_j] for all edges
        # This is simplified; full implementation requires edge-wise computation

        # For efficiency, compute attention using broadcasting
        h_i = h.unsqueeze(2)  # [num_heads, N, 1, out_features]
        h_j = h.unsqueeze(1)  # [num_heads, 1, N, out_features]

        # Concatenate for attention
        h_cat = torch.cat([h_i.expand(-1, -1, N, -1),
                           h_j.expand(-1, N, -1, -1)], dim=-1)
        # [num_heads, N, N, 2*out_features]

        # Compute attention scores
        e = torch.matmul(h_cat, self.a).squeeze(-1)  # [num_heads, N, N]
        e = self.leakyrelu(e)

        # Mask attention to only neighbors (use adj mask)
        adj_dense = adj.to_dense()  # [N, N]
        mask = adj_dense.unsqueeze(0).expand(self.num_heads, -1, -1)  # [num_heads, N, N]

        # Apply mask (set non-neighbors to -inf)
        e = torch.where(mask > 0, e, torch.tensor(-1e9).to(e.device))

        # Softmax to get attention weights
        alpha = F.softmax(e, dim=2)  # [num_heads, N, N]

        if self.dropout is not None:
            alpha = self.dropout(alpha)

        # Aggregate with attention: [num_heads, N, out_features]
        h_out = torch.bmm(alpha, h)

        # Combine heads
        if self.concat:
            # Concatenate heads: [N, num_heads * out_features]
            output = h_out.transpose(0, 1).contiguous().view(N, -1)
        else:
            # Average heads: [N, out_features]
            output = h_out.mean(dim=0)

        if self.bias is not None:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, heads={self.num_heads})"


class MultiLayerGNN(nn.Module):
    """
    Multi-layer GNN with residual connections and layer normalization.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 3,
        gnn_type: str = 'gcn',  # 'gcn', 'sage', 'gat'
        activation: Optional[Callable] = F.relu,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        **kwargs
    ):
        super().__init__()

        self.num_layers = num_layers
        self.use_residual = use_residual

        # Build layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None

        layer_class = {
            'gcn': GCNLayer,
            'sage': GraphSAGELayer,
            'gat': GATLayer
        }[gnn_type.lower()]

        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_features
            out_dim = out_features if i == num_layers - 1 else hidden_features

            layer = layer_class(
                in_dim,
                out_dim,
                activation=activation if i < num_layers - 1 else None,
                dropout=dropout,
                **kwargs
            )
            self.layers.append(layer)

            if use_layer_norm and i < num_layers - 1:
                self.norms.append(nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""

        h = x
        for i, layer in enumerate(self.layers):
            h_new = layer(h, adj)

            # Residual connection (if dimensions match)
            if self.use_residual and h.size(-1) == h_new.size(-1):
                h_new = h_new + h

            # Layer normalization
            if self.norms is not None and i < len(self.norms):
                h_new = self.norms[i](h_new)

            h = h_new

        return h

    def __repr__(self):
        return f"{self.__class__.__name__}(layers={self.num_layers})"
