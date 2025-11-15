#ifndef OMNIGRAPH_SPARSE_GRAPH_HPP
#define OMNIGRAPH_SPARSE_GRAPH_HPP

#include "types.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace omnigraph {

/**
 * @brief Compressed Sparse Row (CSR) graph representation
 *
 * Optimized for fast row access and SpMM operations.
 * Thread-safe for read operations.
 */
class SparseGraph {
public:
    // Constructors
    SparseGraph();
    SparseGraph(IndexType num_nodes, IndexType num_edges);

    // Build from edge list
    void build_from_edges(
        const std::vector<Edge>& edges,
        IndexType num_nodes,
        bool directed = false
    );

    // Build from CSR arrays
    void build_from_csr(
        const std::vector<IndexType>& row_ptr,
        const std::vector<IndexType>& col_idx,
        const std::vector<ValueType>& values,
        IndexType num_nodes
    );

    // Build from PyTorch sparse tensor
    void build_from_torch(const TorchTensor& sparse_tensor);

    // Build from Eigen sparse matrix
    void build_from_eigen(const SparseMatrix& matrix);

    // Accessors
    IndexType num_nodes() const { return num_nodes_; }
    IndexType num_edges() const { return num_edges_; }

    const std::vector<IndexType>& row_ptr() const { return row_ptr_; }
    const std::vector<IndexType>& col_idx() const { return col_idx_; }
    const std::vector<ValueType>& values() const { return values_; }

    // Get neighbors of a node
    std::pair<const IndexType*, const ValueType*> neighbors(IndexType node) const;
    IndexType degree(IndexType node) const;

    // Conversions
    TorchTensor to_torch_sparse() const;
    SparseMatrix to_eigen_sparse() const;
    DenseMatrix to_dense() const;

    // Graph statistics
    GraphStats compute_stats() const;

    // Graph transformations
    SparseGraph transpose() const;
    SparseGraph normalize(const std::string& method = "symmetric") const;
    SparseGraph add_self_loops(ValueType weight = 1.0f) const;

    // Subgraph extraction
    SparseGraph extract_subgraph(const std::vector<IndexType>& nodes) const;

    // Serialization
    void save(const std::string& filename) const;
    void load(const std::string& filename);

    // Validate integrity
    bool is_valid() const;

private:
    IndexType num_nodes_;
    IndexType num_edges_;

    // CSR storage
    std::vector<IndexType> row_ptr_;   // Size: num_nodes + 1
    std::vector<IndexType> col_idx_;   // Size: num_edges
    std::vector<ValueType> values_;    // Size: num_edges

    // Optional metadata
    std::unordered_map<std::string, std::string> metadata_;

    // Helper functions
    void validate_csr() const;
    void sort_neighbors();
};

/**
 * @brief Graph builder utility class
 */
class GraphBuilder {
public:
    static SparseGraph build_knn_graph(
        const DenseMatrix& features,
        int k,
        const std::string& metric = "euclidean",
        bool mutual = false
    );

    static SparseGraph build_knn_graph_torch(
        const TorchTensor& features,
        int k,
        const std::string& metric = "euclidean",
        bool mutual = false
    );

    static SparseGraph build_distance_graph(
        const DenseMatrix& coordinates,
        double radius,
        const std::string& kernel = "gaussian",
        double sigma = 1.0
    );

    static SparseGraph build_correlation_graph(
        const DenseMatrix& data,
        double threshold,
        const std::string& method = "pearson"
    );
};

} // namespace omnigraph

#endif // OMNIGRAPH_SPARSE_GRAPH_HPP
