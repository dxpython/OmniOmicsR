#ifndef OMNIGRAPH_TYPES_HPP
#define OMNIGRAPH_TYPES_HPP

#include <torch/torch.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <cstdint>

namespace omnigraph {

// Type aliases for clarity
using IndexType = int64_t;
using ValueType = float;

// Eigen types
using DenseMatrix = Eigen::MatrixXf;
using DenseVector = Eigen::VectorXf;
using SparseMatrix = Eigen::SparseMatrix<ValueType, Eigen::RowMajor, IndexType>;

// PyTorch tensor types
using TorchTensor = torch::Tensor;
using TorchDevice = torch::Device;

// Graph edge representation
struct Edge {
    IndexType src;
    IndexType dst;
    ValueType weight;

    Edge(IndexType s, IndexType d, ValueType w = 1.0f)
        : src(s), dst(d), weight(w) {}
};

// Graph statistics
struct GraphStats {
    IndexType num_nodes;
    IndexType num_edges;
    double avg_degree;
    double density;
    IndexType max_degree;
    IndexType min_degree;
};

// Tensor conversion utilities
namespace utils {

// Convert Eigen matrix to PyTorch tensor (zero-copy when possible)
inline TorchTensor eigen_to_torch(const DenseMatrix& matrix) {
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU);

    // Create tensor from data (copy)
    auto tensor = torch::from_blob(
        const_cast<float*>(matrix.data()),
        {matrix.rows(), matrix.cols()},
        options
    ).clone();  // Clone to own the memory

    return tensor;
}

// Convert PyTorch tensor to Eigen matrix
inline DenseMatrix torch_to_eigen(const TorchTensor& tensor) {
    TORCH_CHECK(tensor.dim() == 2, "Tensor must be 2D");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, "Tensor must be float32");

    auto cpu_tensor = tensor.to(torch::kCPU).contiguous();
    auto data_ptr = cpu_tensor.data_ptr<float>();
    int64_t rows = cpu_tensor.size(0);
    int64_t cols = cpu_tensor.size(1);

    // Map Eigen matrix to PyTorch data
    return Eigen::Map<const DenseMatrix>(data_ptr, rows, cols);
}

// Extract CSR indices from PyTorch sparse tensor
inline std::tuple<std::vector<IndexType>, std::vector<IndexType>, std::vector<ValueType>>
torch_sparse_to_csr(const TorchTensor& sparse_tensor, IndexType num_nodes) {
    TORCH_CHECK(sparse_tensor.is_sparse(), "Tensor must be sparse");

    auto indices = sparse_tensor._indices().to(torch::kCPU);
    auto values = sparse_tensor._values().to(torch::kCPU);

    auto row_indices = indices[0].accessor<int64_t, 1>();
    auto col_indices = indices[1].accessor<int64_t, 1>();
    auto value_data = values.accessor<float, 1>();

    int64_t nnz = values.size(0);

    // Build edge list first
    std::vector<Edge> edges;
    edges.reserve(nnz);
    for (int64_t i = 0; i < nnz; i++) {
        edges.emplace_back(row_indices[i], col_indices[i], value_data[i]);
    }

    // Sort by row then column
    std::sort(edges.begin(), edges.end(),
        [](const Edge& a, const Edge& b) {
            return a.src < b.src || (a.src == b.src && a.dst < b.dst);
        });

    // Convert to CSR format
    std::vector<IndexType> row_ptr(num_nodes + 1, 0);
    std::vector<IndexType> col_idx;
    std::vector<ValueType> vals;

    col_idx.reserve(nnz);
    vals.reserve(nnz);

    IndexType current_row = 0;
    for (const auto& edge : edges) {
        // Fill row pointers for empty rows
        while (current_row < edge.src) {
            row_ptr[current_row + 1] = col_idx.size();
            current_row++;
        }

        col_idx.push_back(edge.dst);
        vals.push_back(edge.weight);
    }

    // Fill remaining row pointers
    while (current_row < num_nodes) {
        row_ptr[current_row + 1] = col_idx.size();
        current_row++;
    }

    return {row_ptr, col_idx, vals};
}

// Create PyTorch sparse tensor from CSR
inline TorchTensor csr_to_torch_sparse(
    const std::vector<IndexType>& row_ptr,
    const std::vector<IndexType>& col_idx,
    const std::vector<ValueType>& values,
    IndexType num_rows,
    IndexType num_cols
) {
    IndexType nnz = col_idx.size();

    // Build COO format (PyTorch uses COO)
    std::vector<IndexType> rows, cols;
    rows.reserve(nnz);
    cols.reserve(nnz);

    for (IndexType i = 0; i < num_rows; i++) {
        for (IndexType j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            rows.push_back(i);
            cols.push_back(col_idx[j]);
        }
    }

    // Create index tensor [2, nnz]
    auto indices = torch::zeros({2, nnz}, torch::kInt64);
    auto indices_acc = indices.accessor<int64_t, 2>();
    for (int64_t i = 0; i < nnz; i++) {
        indices_acc[0][i] = rows[i];
        indices_acc[1][i] = cols[i];
    }

    // Create values tensor
    auto values_tensor = torch::from_blob(
        const_cast<float*>(values.data()),
        {nnz},
        torch::kFloat32
    ).clone();

    // Create sparse tensor
    return torch::sparse_coo_tensor(
        indices,
        values_tensor,
        {num_rows, num_cols},
        torch::TensorOptions().dtype(torch::kFloat32)
    );
}

} // namespace utils

} // namespace omnigraph

#endif // OMNIGRAPH_TYPES_HPP
