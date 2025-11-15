#ifndef OMNIGRAPH_GRAPH_OPS_HPP
#define OMNIGRAPH_GRAPH_OPS_HPP

#include "sparse_graph.hpp"
#include "types.hpp"

namespace omnigraph {
namespace ops {

/**
 * @brief Sparse matrix-matrix multiplication (SpMM)
 *
 * Computes C = A @ B where A is sparse (CSR) and B is dense
 *
 * @param graph Sparse matrix A in SparseGraph format
 * @param B_dense Dense matrix B (PyTorch tensor or Eigen matrix)
 * @return Dense result C
 */
TorchTensor sparse_mm_torch(
    const SparseGraph& graph,
    const TorchTensor& B_dense
);

DenseMatrix sparse_mm_eigen(
    const SparseGraph& graph,
    const DenseMatrix& B_dense
);

/**
 * @brief Sparse matrix-matrix multiplication with transpose
 *
 * Computes C = A^T @ B
 */
TorchTensor sparse_mm_transpose_torch(
    const SparseGraph& graph,
    const TorchTensor& B_dense
);

/**
 * @brief Graph Laplacian computation
 *
 * Computes normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
 *
 * @param graph Input adjacency matrix
 * @param symmetric If true, compute symmetric normalization; else row normalization
 * @return Laplacian as SparseGraph
 */
SparseGraph compute_laplacian(
    const SparseGraph& graph,
    bool symmetric = true
);

/**
 * @brief Compute degree matrix
 *
 * @param graph Input graph
 * @param out_degree If true, compute out-degree; else in-degree
 * @return Degree vector
 */
DenseVector compute_degree(
    const SparseGraph& graph,
    bool out_degree = true
);

TorchTensor compute_degree_torch(
    const SparseGraph& graph,
    bool out_degree = true
);

/**
 * @brief Normalize adjacency matrix
 *
 * @param graph Input graph
 * @param method "symmetric" (D^(-1/2) A D^(-1/2)) or "row" (D^(-1) A)
 * @return Normalized graph
 */
SparseGraph normalize_adjacency(
    const SparseGraph& graph,
    const std::string& method = "symmetric"
);

/**
 * @brief Add self-loops to graph
 *
 * @param graph Input graph
 * @param weight Weight of self-loops (default: 1.0)
 * @return Graph with self-loops
 */
SparseGraph add_self_loops(
    const SparseGraph& graph,
    ValueType weight = 1.0f
);

/**
 * @brief GCN aggregation (with symmetric normalization)
 *
 * Computes: Â = D̃^(-1/2) (A + I) D̃^(-1/2)
 * where D̃ is the degree matrix of A + I
 *
 * @param graph Input adjacency matrix
 * @return Normalized adjacency with self-loops
 */
SparseGraph gcn_norm(const SparseGraph& graph);

/**
 * @brief Graph convolution operation
 *
 * Computes: H_out = σ(Â @ H_in @ W)
 * where Â is normalized adjacency
 *
 * @param graph Normalized adjacency matrix
 * @param features Input node features [N, D_in]
 * @param weight Weight matrix [D_in, D_out] (PyTorch tensor)
 * @param bias Bias vector [D_out] (optional)
 * @param activation Activation function name ("relu", "none", etc.)
 * @return Output features [N, D_out]
 */
TorchTensor graph_conv(
    const SparseGraph& graph,
    const TorchTensor& features,
    const TorchTensor& weight,
    const TorchTensor& bias = {},
    const std::string& activation = "none"
);

/**
 * @brief Compute graph smoothness loss
 *
 * Computes: tr(Z^T L Z) where L is the Laplacian
 *
 * @param laplacian Graph Laplacian
 * @param embeddings Node embeddings [N, D]
 * @return Smoothness loss (scalar)
 */
ValueType graph_smoothness_loss(
    const SparseGraph& laplacian,
    const TorchTensor& embeddings
);

TorchTensor graph_smoothness_loss_torch(
    const SparseGraph& laplacian,
    const TorchTensor& embeddings
);

/**
 * @brief PageRank computation
 *
 * Computes PageRank scores for all nodes
 *
 * @param graph Input graph
 * @param damping Damping factor (default: 0.85)
 * @param max_iter Maximum iterations (default: 100)
 * @param tol Convergence tolerance (default: 1e-6)
 * @return PageRank scores [N]
 */
DenseVector compute_pagerank(
    const SparseGraph& graph,
    double damping = 0.85,
    int max_iter = 100,
    double tol = 1e-6
);

/**
 * @brief Random walk sampling
 *
 * Performs random walks starting from seed nodes
 *
 * @param graph Input graph
 * @param start_nodes Seed nodes [num_seeds]
 * @param walk_length Length of each walk
 * @param num_walks Number of walks per seed
 * @return Walk paths [num_seeds * num_walks, walk_length]
 */
TorchTensor random_walk(
    const SparseGraph& graph,
    const TorchTensor& start_nodes,
    int walk_length,
    int num_walks = 1
);

/**
 * @brief Matrix power computation (A^k)
 *
 * Efficiently computes k-th power of sparse matrix
 *
 * @param graph Input matrix A
 * @param k Exponent
 * @return A^k as SparseGraph
 */
SparseGraph matrix_power(
    const SparseGraph& graph,
    int k
);

/**
 * @brief Compute k-hop neighbors
 *
 * Find all nodes within k hops of seed nodes
 *
 * @param graph Input graph
 * @param seed_nodes Starting nodes
 * @param k Number of hops
 * @return Set of k-hop neighbors
 */
std::vector<IndexType> k_hop_neighbors(
    const SparseGraph& graph,
    const std::vector<IndexType>& seed_nodes,
    int k
);

} // namespace ops
} // namespace omnigraph

#endif // OMNIGRAPH_GRAPH_OPS_HPP
