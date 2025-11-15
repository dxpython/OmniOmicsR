#include "omnigraph/sparse_graph.hpp"
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <cmath>

namespace omnigraph {

// ==================== SparseGraph Implementation ====================

SparseGraph::SparseGraph()
    : num_nodes_(0), num_edges_(0) {}

SparseGraph::SparseGraph(IndexType num_nodes, IndexType num_edges)
    : num_nodes_(num_nodes), num_edges_(num_edges) {
    row_ptr_.resize(num_nodes + 1, 0);
    col_idx_.reserve(num_edges);
    values_.reserve(num_edges);
}

void SparseGraph::build_from_edges(
    const std::vector<Edge>& edges,
    IndexType num_nodes,
    bool directed
) {
    num_nodes_ = num_nodes;

    // Copy and possibly duplicate edges for undirected graph
    std::vector<Edge> all_edges = edges;
    if (!directed) {
        size_t original_size = edges.size();
        all_edges.reserve(original_size * 2);
        for (size_t i = 0; i < original_size; i++) {
            if (edges[i].src != edges[i].dst) {  // Skip self-loops
                all_edges.emplace_back(edges[i].dst, edges[i].src, edges[i].weight);
            }
        }
    }

    // Sort edges by (src, dst)
    std::sort(all_edges.begin(), all_edges.end(),
        [](const Edge& a, const Edge& b) {
            return a.src < b.src || (a.src == b.src && a.dst < b.dst);
        });

    // Remove duplicates (keep first occurrence)
    all_edges.erase(
        std::unique(all_edges.begin(), all_edges.end(),
            [](const Edge& a, const Edge& b) {
                return a.src == b.src && a.dst == b.dst;
            }),
        all_edges.end()
    );

    num_edges_ = all_edges.size();

    // Build CSR
    row_ptr_.clear();
    row_ptr_.resize(num_nodes + 1, 0);
    col_idx_.clear();
    col_idx_.reserve(num_edges_);
    values_.clear();
    values_.reserve(num_edges_);

    IndexType current_row = 0;
    for (const auto& edge : all_edges) {
        // Fill row pointers for empty rows
        while (current_row < edge.src) {
            row_ptr_[current_row + 1] = col_idx_.size();
            current_row++;
        }

        col_idx_.push_back(edge.dst);
        values_.push_back(edge.weight);
    }

    // Fill remaining row pointers
    while (current_row < num_nodes_) {
        row_ptr_[current_row + 1] = col_idx_.size();
        current_row++;
    }

    validate_csr();
}

void SparseGraph::build_from_csr(
    const std::vector<IndexType>& row_ptr,
    const std::vector<IndexType>& col_idx,
    const std::vector<ValueType>& values,
    IndexType num_nodes
) {
    num_nodes_ = num_nodes;
    num_edges_ = col_idx.size();

    row_ptr_ = row_ptr;
    col_idx_ = col_idx;
    values_ = values;

    validate_csr();
}

void SparseGraph::build_from_torch(const TorchTensor& sparse_tensor) {
    TORCH_CHECK(sparse_tensor.is_sparse(), "Input must be sparse tensor");

    num_nodes_ = sparse_tensor.size(0);

    auto csr_data = utils::torch_sparse_to_csr(sparse_tensor, num_nodes_);
    row_ptr_ = std::get<0>(csr_data);
    col_idx_ = std::get<1>(csr_data);
    values_ = std::get<2>(csr_data);

    num_edges_ = col_idx_.size();
    validate_csr();
}

void SparseGraph::build_from_eigen(const SparseMatrix& matrix) {
    num_nodes_ = matrix.rows();
    num_edges_ = matrix.nonZeros();

    row_ptr_.resize(num_nodes_ + 1);
    col_idx_.resize(num_edges_);
    values_.resize(num_edges_);

    // Copy data from Eigen sparse matrix
    const IndexType* outer_ptr = matrix.outerIndexPtr();
    const IndexType* inner_ptr = matrix.innerIndexPtr();
    const ValueType* value_ptr = matrix.valuePtr();

    std::copy(outer_ptr, outer_ptr + num_nodes_ + 1, row_ptr_.begin());
    std::copy(inner_ptr, inner_ptr + num_edges_, col_idx_.begin());
    std::copy(value_ptr, value_ptr + num_edges_, values_.begin());

    validate_csr();
}

std::pair<const IndexType*, const ValueType*>
SparseGraph::neighbors(IndexType node) const {
    if (node < 0 || node >= num_nodes_) {
        throw std::out_of_range("Node index out of range");
    }

    const IndexType* col_start = col_idx_.data() + row_ptr_[node];
    const ValueType* val_start = values_.data() + row_ptr_[node];

    return {col_start, val_start};
}

IndexType SparseGraph::degree(IndexType node) const {
    if (node < 0 || node >= num_nodes_) {
        throw std::out_of_range("Node index out of range");
    }
    return row_ptr_[node + 1] - row_ptr_[node];
}

TorchTensor SparseGraph::to_torch_sparse() const {
    return utils::csr_to_torch_sparse(
        row_ptr_, col_idx_, values_,
        num_nodes_, num_nodes_
    );
}

SparseMatrix SparseGraph::to_eigen_sparse() const {
    SparseMatrix matrix(num_nodes_, num_nodes_);
    matrix.reserve(num_edges_);

    for (IndexType i = 0; i < num_nodes_; i++) {
        for (IndexType j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
            matrix.insert(i, col_idx_[j]) = values_[j];
        }
    }

    matrix.makeCompressed();
    return matrix;
}

DenseMatrix SparseGraph::to_dense() const {
    DenseMatrix dense = DenseMatrix::Zero(num_nodes_, num_nodes_);

    for (IndexType i = 0; i < num_nodes_; i++) {
        for (IndexType j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
            dense(i, col_idx_[j]) = values_[j];
        }
    }

    return dense;
}

GraphStats SparseGraph::compute_stats() const {
    GraphStats stats;
    stats.num_nodes = num_nodes_;
    stats.num_edges = num_edges_;

    if (num_nodes_ > 0) {
        stats.avg_degree = static_cast<double>(num_edges_) / num_nodes_;
        stats.density = static_cast<double>(num_edges_) /
                       (static_cast<double>(num_nodes_) * num_nodes_);

        // Compute max/min degree
        stats.max_degree = 0;
        stats.min_degree = std::numeric_limits<IndexType>::max();

        for (IndexType i = 0; i < num_nodes_; i++) {
            IndexType deg = degree(i);
            stats.max_degree = std::max(stats.max_degree, deg);
            stats.min_degree = std::min(stats.min_degree, deg);
        }
    } else {
        stats.avg_degree = 0.0;
        stats.density = 0.0;
        stats.max_degree = 0;
        stats.min_degree = 0;
    }

    return stats;
}

SparseGraph SparseGraph::transpose() const {
    // Build edge list
    std::vector<Edge> edges;
    edges.reserve(num_edges_);

    for (IndexType i = 0; i < num_nodes_; i++) {
        for (IndexType j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
            // Swap src and dst
            edges.emplace_back(col_idx_[j], i, values_[j]);
        }
    }

    SparseGraph transposed;
    transposed.build_from_edges(edges, num_nodes_, true);
    return transposed;
}

SparseGraph SparseGraph::normalize(const std::string& method) const {
    // Compute degree
    std::vector<ValueType> degree_vec(num_nodes_, 0.0f);

    for (IndexType i = 0; i < num_nodes_; i++) {
        for (IndexType j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
            degree_vec[i] += values_[j];
        }
    }

    // Build normalized graph
    std::vector<Edge> edges;
    edges.reserve(num_edges_);

    if (method == "symmetric") {
        // D^(-1/2) A D^(-1/2)
        for (IndexType i = 0; i < num_nodes_; i++) {
            ValueType d_i_sqrt = std::sqrt(degree_vec[i] + 1e-6f);

            for (IndexType j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
                IndexType dst = col_idx_[j];
                ValueType d_j_sqrt = std::sqrt(degree_vec[dst] + 1e-6f);
                ValueType norm_weight = values_[j] / (d_i_sqrt * d_j_sqrt);

                edges.emplace_back(i, dst, norm_weight);
            }
        }
    } else if (method == "row") {
        // D^(-1) A
        for (IndexType i = 0; i < num_nodes_; i++) {
            ValueType d_i = degree_vec[i] + 1e-6f;

            for (IndexType j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
                IndexType dst = col_idx_[j];
                ValueType norm_weight = values_[j] / d_i;

                edges.emplace_back(i, dst, norm_weight);
            }
        }
    } else {
        throw std::invalid_argument("Invalid normalization method");
    }

    SparseGraph normalized;
    normalized.build_from_edges(edges, num_nodes_, true);
    return normalized;
}

SparseGraph SparseGraph::add_self_loops(ValueType weight) const {
    std::vector<Edge> edges;
    edges.reserve(num_edges_ + num_nodes_);

    // Add existing edges
    for (IndexType i = 0; i < num_nodes_; i++) {
        for (IndexType j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
            edges.emplace_back(i, col_idx_[j], values_[j]);
        }
    }

    // Add self-loops
    for (IndexType i = 0; i < num_nodes_; i++) {
        edges.emplace_back(i, i, weight);
    }

    SparseGraph result;
    result.build_from_edges(edges, num_nodes_, true);
    return result;
}

void SparseGraph::validate_csr() const {
    if (row_ptr_.size() != static_cast<size_t>(num_nodes_ + 1)) {
        throw std::runtime_error("Invalid CSR: row_ptr size mismatch");
    }

    if (col_idx_.size() != static_cast<size_t>(num_edges_) ||
        values_.size() != static_cast<size_t>(num_edges_)) {
        throw std::runtime_error("Invalid CSR: column/value size mismatch");
    }

    if (row_ptr_[0] != 0 || row_ptr_[num_nodes_] != num_edges_) {
        throw std::runtime_error("Invalid CSR: row_ptr boundaries");
    }

    // Check monotonicity
    for (IndexType i = 0; i < num_nodes_; i++) {
        if (row_ptr_[i] > row_ptr_[i + 1]) {
            throw std::runtime_error("Invalid CSR: row_ptr not monotonic");
        }

        // Check column indices are valid
        for (IndexType j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
            if (col_idx_[j] < 0 || col_idx_[j] >= num_nodes_) {
                throw std::runtime_error("Invalid CSR: column index out of range");
            }
        }
    }
}

bool SparseGraph::is_valid() const {
    try {
        validate_csr();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// ==================== GraphBuilder Implementation ====================

SparseGraph GraphBuilder::build_knn_graph(
    const DenseMatrix& features,
    int k,
    const std::string& metric,
    bool mutual
) {
    IndexType num_nodes = features.rows();
    std::vector<Edge> edges;
    edges.reserve(num_nodes * k);

    // Simple k-NN (can be optimized with KD-tree for large datasets)
    for (IndexType i = 0; i < num_nodes; i++) {
        // Compute distances to all other nodes
        std::vector<std::pair<ValueType, IndexType>> distances;
        distances.reserve(num_nodes - 1);

        for (IndexType j = 0; j < num_nodes; j++) {
            if (i == j) continue;

            ValueType dist;
            if (metric == "euclidean") {
                dist = (features.row(i) - features.row(j)).norm();
            } else if (metric == "cosine") {
                ValueType dot = features.row(i).dot(features.row(j));
                ValueType norm_i = features.row(i).norm();
                ValueType norm_j = features.row(j).norm();
                dist = 1.0f - dot / (norm_i * norm_j + 1e-8f);
            } else {
                throw std::invalid_argument("Unknown metric");
            }

            distances.emplace_back(dist, j);
        }

        // Sort by distance and take k nearest
        std::partial_sort(
            distances.begin(),
            distances.begin() + std::min(k, static_cast<int>(distances.size())),
            distances.end()
        );

        // Add edges
        for (int kk = 0; kk < std::min(k, static_cast<int>(distances.size())); kk++) {
            ValueType weight = 1.0f / (distances[kk].first + 1e-6f);  // Convert distance to similarity
            edges.emplace_back(i, distances[kk].second, weight);
        }
    }

    SparseGraph graph;
    graph.build_from_edges(edges, num_nodes, mutual ? false : true);
    return graph;
}

} // namespace omnigraph
