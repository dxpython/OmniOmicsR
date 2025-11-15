#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include "omnigraph/sparse_graph.hpp"
#include "omnigraph/graph_ops.hpp"
#include "omnigraph/message_passing.hpp"
#include "omnigraph/sampling.hpp"

namespace py = pybind11;
using namespace omnigraph;

PYBIND11_MODULE(omnigraph_cpp, m) {
    m.doc() = "OmniGraphDiff C++ Backend - High-performance graph operations";

    // ==================== Types and Enums ====================

    py::class_<Edge>(m, "Edge")
        .def(py::init<IndexType, IndexType, ValueType>(),
             py::arg("src"), py::arg("dst"), py::arg("weight") = 1.0f)
        .def_readwrite("src", &Edge::src)
        .def_readwrite("dst", &Edge::dst)
        .def_readwrite("weight", &Edge::weight);

    py::class_<GraphStats>(m, "GraphStats")
        .def_readonly("num_nodes", &GraphStats::num_nodes)
        .def_readonly("num_edges", &GraphStats::num_edges)
        .def_readonly("avg_degree", &GraphStats::avg_degree)
        .def_readonly("density", &GraphStats::density)
        .def_readonly("max_degree", &GraphStats::max_degree)
        .def_readonly("min_degree", &GraphStats::min_degree)
        .def("__repr__", [](const GraphStats& s) {
            return "GraphStats(nodes=" + std::to_string(s.num_nodes) +
                   ", edges=" + std::to_string(s.num_edges) +
                   ", avg_degree=" + std::to_string(s.avg_degree) + ")";
        });

    py::enum_<message_passing::AggregationType>(m, "AggregationType")
        .value("SUM", message_passing::AggregationType::SUM)
        .value("MEAN", message_passing::AggregationType::MEAN)
        .value("MAX", message_passing::AggregationType::MAX)
        .value("MIN", message_passing::AggregationType::MIN)
        .value("STD", message_passing::AggregationType::STD)
        .export_values();

    // ==================== SparseGraph ====================

    py::class_<SparseGraph>(m, "SparseGraph")
        .def(py::init<>())
        .def(py::init<IndexType, IndexType>(),
             py::arg("num_nodes"), py::arg("num_edges"))

        // Build methods
        .def("build_from_edges",
             &SparseGraph::build_from_edges,
             py::arg("edges"),
             py::arg("num_nodes"),
             py::arg("directed") = false,
             "Build graph from edge list")

        .def("build_from_torch",
             &SparseGraph::build_from_torch,
             py::arg("sparse_tensor"),
             "Build graph from PyTorch sparse tensor")

        // Accessors
        .def_property_readonly("num_nodes", &SparseGraph::num_nodes)
        .def_property_readonly("num_edges", &SparseGraph::num_edges)
        .def("degree", &SparseGraph::degree, py::arg("node"))

        // Conversions
        .def("to_torch_sparse", &SparseGraph::to_torch_sparse,
             "Convert to PyTorch sparse tensor")
        .def("to_dense", &SparseGraph::to_dense,
             "Convert to dense matrix (Eigen)")

        // Statistics
        .def("compute_stats", &SparseGraph::compute_stats,
             "Compute graph statistics")

        // Transformations
        .def("transpose", &SparseGraph::transpose,
             "Transpose graph")
        .def("normalize", &SparseGraph::normalize,
             py::arg("method") = "symmetric",
             "Normalize adjacency matrix")
        .def("add_self_loops", &SparseGraph::add_self_loops,
             py::arg("weight") = 1.0f,
             "Add self-loops to graph")

        // Validation
        .def("is_valid", &SparseGraph::is_valid,
             "Check if graph is valid")

        .def("__repr__", [](const SparseGraph& g) {
            return "SparseGraph(nodes=" + std::to_string(g.num_nodes()) +
                   ", edges=" + std::to_string(g.num_edges()) + ")";
        });

    // ==================== GraphBuilder ====================

    py::class_<GraphBuilder>(m, "GraphBuilder")
        .def_static("build_knn_graph_torch",
                    &GraphBuilder::build_knn_graph_torch,
                    py::arg("features"),
                    py::arg("k"),
                    py::arg("metric") = "euclidean",
                    py::arg("mutual") = false,
                    "Build k-NN graph from PyTorch features tensor")

        .def_static("build_distance_graph",
                    &GraphBuilder::build_distance_graph,
                    py::arg("coordinates"),
                    py::arg("radius"),
                    py::arg("kernel") = "gaussian",
                    py::arg("sigma") = 1.0,
                    "Build distance-based graph from spatial coordinates");

    // ==================== Graph Operations ====================

    m.def("sparse_mm", &ops::sparse_mm_torch,
          py::arg("graph"), py::arg("dense_matrix"),
          "Sparse matrix-matrix multiplication (A @ B)");

    m.def("sparse_mm_transpose", &ops::sparse_mm_transpose_torch,
          py::arg("graph"), py::arg("dense_matrix"),
          "Sparse matrix-matrix multiplication with transpose (A^T @ B)");

    m.def("compute_laplacian", &ops::compute_laplacian,
          py::arg("graph"), py::arg("symmetric") = true,
          "Compute graph Laplacian");

    m.def("compute_degree", &ops::compute_degree_torch,
          py::arg("graph"), py::arg("out_degree") = true,
          "Compute node degrees");

    m.def("normalize_adjacency", &ops::normalize_adjacency,
          py::arg("graph"), py::arg("method") = "symmetric",
          "Normalize adjacency matrix");

    m.def("gcn_norm", &ops::gcn_norm,
          py::arg("graph"),
          "GCN normalization: D̃^(-1/2) (A + I) D̃^(-1/2)");

    m.def("graph_conv", &ops::graph_conv,
          py::arg("graph"),
          py::arg("features"),
          py::arg("weight"),
          py::arg("bias") = torch::Tensor(),
          py::arg("activation") = "none",
          "Graph convolution operation");

    m.def("graph_smoothness_loss", &ops::graph_smoothness_loss_torch,
          py::arg("laplacian"), py::arg("embeddings"),
          "Compute graph smoothness loss: tr(Z^T L Z)");

    m.def("random_walk", &ops::random_walk,
          py::arg("graph"),
          py::arg("start_nodes"),
          py::arg("walk_length"),
          py::arg("num_walks") = 1,
          "Perform random walks from seed nodes");

    // ==================== Message Passing ====================

    py::class_<message_passing::MessagePassing>(m, "MessagePassing")
        .def_static("aggregate_sum",
                    &message_passing::MessagePassing::aggregate_sum,
                    py::arg("graph"), py::arg("messages"),
                    "Sum aggregation")

        .def_static("aggregate_mean",
                    &message_passing::MessagePassing::aggregate_mean,
                    py::arg("graph"), py::arg("messages"),
                    "Mean aggregation")

        .def_static("aggregate_max",
                    &message_passing::MessagePassing::aggregate_max,
                    py::arg("graph"), py::arg("messages"),
                    "Max aggregation")

        .def_static("sage_aggregation",
                    &message_passing::MessagePassing::sage_aggregation,
                    py::arg("graph"),
                    py::arg("node_features"),
                    py::arg("neighbor_agg") = message_passing::AggregationType::MEAN,
                    "GraphSAGE-style aggregation");

    py::class_<message_passing::BatchMessagePassing>(m, "BatchMessagePassing")
        .def_static("global_pool",
                    &message_passing::BatchMessagePassing::global_pool,
                    py::arg("node_features"),
                    py::arg("batch_indices"),
                    py::arg("num_graphs"),
                    py::arg("pooling") = "mean",
                    "Global graph pooling");

    // ==================== Sampling ====================

    py::class_<sampling::NeighborSampler>(m, "NeighborSampler")
        .def(py::init<int>(), py::arg("seed") = 42)
        .def("sample", &sampling::NeighborSampler::sample,
             py::arg("graph"),
             py::arg("seed_nodes"),
             py::arg("num_neighbors"),
             py::arg("replace") = false,
             "Sample multi-hop neighborhood")
        .def("sample_layer", &sampling::NeighborSampler::sample_layer,
             py::arg("graph"),
             py::arg("nodes"),
             py::arg("num_samples"),
             py::arg("replace") = false,
             "Sample neighbors for a single layer");

    py::class_<sampling::ClusterSampler>(m, "ClusterSampler")
        .def(py::init<int>(), py::arg("seed") = 42)
        .def("partition", &sampling::ClusterSampler::partition,
             py::arg("graph"),
             py::arg("num_clusters"),
             py::arg("method") = "random",
             "Partition graph into clusters")
        .def("sample_clusters", &sampling::ClusterSampler::sample_clusters,
             py::arg("graph"),
             py::arg("cluster_assignments"),
             py::arg("batch_size"),
             "Sample a batch of clusters");

    py::enum_<sampling::GraphSAINTSampler::SamplingMode>(m, "SamplingMode")
        .value("NODE", sampling::GraphSAINTSampler::SamplingMode::NODE)
        .value("EDGE", sampling::GraphSAINTSampler::SamplingMode::EDGE)
        .value("RANDOM_WALK", sampling::GraphSAINTSampler::SamplingMode::RANDOM_WALK)
        .export_values();

    py::class_<sampling::GraphSAINTSampler>(m, "GraphSAINTSampler")
        .def(py::init<sampling::GraphSAINTSampler::SamplingMode, int>(),
             py::arg("mode") = sampling::GraphSAINTSampler::SamplingMode::NODE,
             py::arg("seed") = 42)
        .def("sample", &sampling::GraphSAINTSampler::sample,
             py::arg("graph"),
             py::arg("budget"),
             py::arg("num_subgraphs") = 1,
             "Sample subgraphs");

    py::class_<sampling::NegativeSampler>(m, "NegativeSampler")
        .def(py::init<int>(), py::arg("seed") = 42)
        .def("sample_negative_edges",
             &sampling::NegativeSampler::sample_negative_edges,
             py::arg("graph"),
             py::arg("num_neg_samples"),
             py::arg("mode") = "uniform",
             "Sample negative edges")
        .def("sample_negative_nodes",
             &sampling::NegativeSampler::sample_negative_nodes,
             py::arg("num_nodes"),
             py::arg("positive_nodes"),
             py::arg("num_negatives"),
             "Sample negative nodes for contrastive learning");

    // ==================== Version Info ====================

    m.attr("__version__") = "0.1.0";
    m.attr("__author__") = "OmniGraphDiff Team";

    // Print compilation info
    m.def("get_build_info", []() {
        py::dict info;
        info["torch_version"] = TORCH_VERSION;
        info["eigen_version"] = std::to_string(EIGEN_WORLD_VERSION) + "." +
                                std::to_string(EIGEN_MAJOR_VERSION) + "." +
                                std::to_string(EIGEN_MINOR_VERSION);
        #ifdef _OPENMP
            info["openmp"] = true;
        #else
            info["openmp"] = false;
        #endif
        info["build_type"] =
        #ifdef NDEBUG
            "Release";
        #else
            "Debug";
        #endif
        return info;
    }, "Get compilation and build information");
}
