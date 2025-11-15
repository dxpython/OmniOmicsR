#ifndef OMNIGRAPH_SAMPLING_HPP
#define OMNIGRAPH_SAMPLING_HPP

#include "sparse_graph.hpp"
#include "types.hpp"
#include <random>
#include <unordered_set>

namespace omnigraph {
namespace sampling {

/**
 * @brief Sampling strategies for mini-batch GNN training
 */

/**
 * @brief Neighbor sampler for GraphSAGE-style sampling
 *
 * Samples a fixed number of neighbors per node per layer
 */
class NeighborSampler {
public:
    NeighborSampler(int seed = 42);

    /**
     * @brief Sample multi-hop neighborhood
     *
     * @param graph Input graph
     * @param seed_nodes Starting nodes [num_seeds]
     * @param num_neighbors Number of neighbors to sample per layer [num_layers]
     * @param replace Sample with replacement
     * @return Tuple of (sampled_nodes, sampled_edges, mapping)
     */
    std::tuple<TorchTensor, TorchTensor, TorchTensor> sample(
        const SparseGraph& graph,
        const TorchTensor& seed_nodes,
        const std::vector<int>& num_neighbors,
        bool replace = false
    );

    /**
     * @brief Sample neighbors for a single layer
     *
     * @param graph Input graph
     * @param nodes Current frontier nodes
     * @param num_samples Number of neighbors to sample per node
     * @param replace Sample with replacement
     * @return Sampled neighbors [num_nodes * num_samples]
     */
    TorchTensor sample_layer(
        const SparseGraph& graph,
        const TorchTensor& nodes,
        int num_samples,
        bool replace = false
    );

private:
    std::mt19937 rng_;
};

/**
 * @brief ClusterGCN sampling
 *
 * Partitions graph into clusters and samples clusters as mini-batches
 */
class ClusterSampler {
public:
    ClusterSampler(int seed = 42);

    /**
     * @brief Partition graph into clusters
     *
     * @param graph Input graph
     * @param num_clusters Number of clusters
     * @param method "metis", "louvain", or "random"
     * @return Cluster assignment [num_nodes]
     */
    TorchTensor partition(
        const SparseGraph& graph,
        int num_clusters,
        const std::string& method = "random"
    );

    /**
     * @brief Sample a batch of clusters
     *
     * @param graph Input graph
     * @param cluster_assignments Node-to-cluster mapping
     * @param batch_size Number of clusters per batch
     * @return Subgraph containing selected clusters
     */
    std::tuple<SparseGraph, TorchTensor> sample_clusters(
        const SparseGraph& graph,
        const TorchTensor& cluster_assignments,
        int batch_size
    );

private:
    std::mt19937 rng_;

    TorchTensor partition_random(const SparseGraph& graph, int num_clusters);
    TorchTensor partition_metis(const SparseGraph& graph, int num_clusters);
};

/**
 * @brief GraphSAINT sampling
 *
 * Samples subgraphs via random walks, node sampling, or edge sampling
 */
class GraphSAINTSampler {
public:
    enum class SamplingMode {
        NODE,         // Node sampling
        EDGE,         // Edge sampling
        RANDOM_WALK   // Random walk sampling
    };

    GraphSAINTSampler(SamplingMode mode = SamplingMode::NODE, int seed = 42);

    /**
     * @brief Sample a subgraph
     *
     * @param graph Input graph
     * @param budget Sampling budget (num_nodes, num_edges, or walk_length)
     * @param num_subgraphs Number of subgraphs to sample
     * @return Vector of sampled subgraphs
     */
    std::vector<std::tuple<SparseGraph, TorchTensor>> sample(
        const SparseGraph& graph,
        int budget,
        int num_subgraphs = 1
    );

private:
    SamplingMode mode_;
    std::mt19937 rng_;

    std::tuple<SparseGraph, TorchTensor> sample_nodes(
        const SparseGraph& graph,
        int num_nodes
    );

    std::tuple<SparseGraph, TorchTensor> sample_edges(
        const SparseGraph& graph,
        int num_edges
    );

    std::tuple<SparseGraph, TorchTensor> sample_random_walk(
        const SparseGraph& graph,
        int walk_length,
        int num_walks
    );
};

/**
 * @brief Negative sampling for contrastive learning
 */
class NegativeSampler {
public:
    NegativeSampler(int seed = 42);

    /**
     * @brief Sample negative edges (non-existent edges)
     *
     * @param graph Input graph
     * @param num_neg_samples Number of negative samples per positive edge
     * @param mode "uniform" or "degree-weighted"
     * @return Negative edge pairs [num_neg_samples, 2]
     */
    TorchTensor sample_negative_edges(
        const SparseGraph& graph,
        int num_neg_samples,
        const std::string& mode = "uniform"
    );

    /**
     * @brief Sample negative nodes for contrastive learning
     *
     * @param num_nodes Total number of nodes
     * @param positive_nodes Positive node indices
     * @param num_negatives Number of negatives per positive
     * @return Negative node indices
     */
    TorchTensor sample_negative_nodes(
        int num_nodes,
        const TorchTensor& positive_nodes,
        int num_negatives
    );

private:
    std::mt19937 rng_;
};

/**
 * @brief Layer-wise sampling for deep GNNs
 *
 * FastGCN-style layer-wise importance sampling
 */
class LayerWiseSampler {
public:
    LayerWiseSampler(int seed = 42);

    /**
     * @brief Sample nodes for each layer based on importance
     *
     * @param graph Input graph
     * @param num_layers Number of GNN layers
     * @param samples_per_layer Number of nodes to sample per layer
     * @param importance_type "uniform", "degree", or "pagerank"
     * @return Sampled nodes per layer
     */
    std::vector<TorchTensor> sample_layers(
        const SparseGraph& graph,
        int num_layers,
        const std::vector<int>& samples_per_layer,
        const std::string& importance_type = "degree"
    );

private:
    std::mt19937 rng_;

    DenseVector compute_importance(
        const SparseGraph& graph,
        const std::string& type
    );
};

/**
 * @brief Utility functions for sampling
 */
namespace utils {

/**
 * @brief Reservoir sampling
 *
 * Sample k items from stream of n items uniformly
 */
std::vector<IndexType> reservoir_sample(
    const std::vector<IndexType>& items,
    int k,
    std::mt19937& rng
);

/**
 * @brief Weighted sampling without replacement
 *
 * @param weights Sampling weights [N]
 * @param k Number of samples
 * @param rng Random number generator
 * @return Sampled indices
 */
std::vector<IndexType> weighted_sample(
    const DenseVector& weights,
    int k,
    std::mt19937& rng
);

/**
 * @brief Extract induced subgraph
 *
 * @param graph Original graph
 * @param nodes Subset of nodes
 * @return Subgraph and node mapping
 */
std::tuple<SparseGraph, std::unordered_map<IndexType, IndexType>>
extract_induced_subgraph(
    const SparseGraph& graph,
    const std::unordered_set<IndexType>& nodes
);

/**
 * @brief Convert node set to tensor
 */
TorchTensor node_set_to_tensor(
    const std::unordered_set<IndexType>& nodes
);

} // namespace utils

} // namespace sampling
} // namespace omnigraph

#endif // OMNIGRAPH_SAMPLING_HPP
