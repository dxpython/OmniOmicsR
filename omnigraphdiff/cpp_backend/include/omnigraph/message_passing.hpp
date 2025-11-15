#ifndef OMNIGRAPH_MESSAGE_PASSING_HPP
#define OMNIGRAPH_MESSAGE_PASSING_HPP

#include "sparse_graph.hpp"
#include "types.hpp"
#include <functional>

namespace omnigraph {
namespace message_passing {

/**
 * @brief Message passing aggregation types
 */
enum class AggregationType {
    SUM,
    MEAN,
    MAX,
    MIN,
    STD
};

/**
 * @brief Generic message passing framework
 *
 * Implements the message passing paradigm:
 * h_i' = UPDATE(h_i, AGGREGATE({MESSAGE(h_i, h_j, e_ij) : j ∈ N(i)}))
 */
class MessagePassing {
public:
    /**
     * @brief Aggregate messages from neighbors
     *
     * @param graph Graph structure
     * @param node_features Node features [N, D]
     * @param edge_features Edge features [E, D_edge] (optional)
     * @param aggregation Aggregation type
     * @return Aggregated messages [N, D]
     */
    static TorchTensor aggregate(
        const SparseGraph& graph,
        const TorchTensor& node_features,
        const TorchTensor& edge_features,
        AggregationType aggregation = AggregationType::MEAN
    );

    /**
     * @brief Sum aggregation (most efficient)
     */
    static TorchTensor aggregate_sum(
        const SparseGraph& graph,
        const TorchTensor& messages
    );

    /**
     * @brief Mean aggregation
     */
    static TorchTensor aggregate_mean(
        const SparseGraph& graph,
        const TorchTensor& messages
    );

    /**
     * @brief Max aggregation (pooling)
     */
    static TorchTensor aggregate_max(
        const SparseGraph& graph,
        const TorchTensor& messages
    );

    /**
     * @brief GraphSAGE-style aggregation
     *
     * Concatenates self-features with aggregated neighbor features
     */
    static TorchTensor sage_aggregation(
        const SparseGraph& graph,
        const TorchTensor& node_features,
        AggregationType neighbor_agg = AggregationType::MEAN
    );

    /**
     * @brief GAT-style attention aggregation
     *
     * @param graph Graph structure
     * @param node_features Node features [N, D]
     * @param attention_weights Attention scores [E] (pre-computed)
     * @return Attention-weighted aggregation [N, D]
     */
    static TorchTensor attention_aggregation(
        const SparseGraph& graph,
        const TorchTensor& node_features,
        const TorchTensor& attention_weights
    );

    /**
     * @brief Edge-conditioned convolution
     *
     * Messages are transformed by edge features
     */
    static TorchTensor edge_conv(
        const SparseGraph& graph,
        const TorchTensor& node_features,
        const TorchTensor& edge_features,
        const TorchTensor& weight_matrix
    );
};

/**
 * @brief Batch message passing for mini-batch training
 */
class BatchMessagePassing {
public:
    /**
     * @brief Batch-wise aggregation with support for variable-size graphs
     *
     * @param graphs Vector of graphs (one per sample in batch)
     * @param node_features Batched node features [total_nodes, D]
     * @param batch_indices Node-to-graph mapping [total_nodes]
     * @param aggregation Aggregation type
     * @return Aggregated messages [total_nodes, D]
     */
    static TorchTensor batch_aggregate(
        const std::vector<SparseGraph>& graphs,
        const TorchTensor& node_features,
        const TorchTensor& batch_indices,
        AggregationType aggregation = AggregationType::MEAN
    );

    /**
     * @brief Global pooling over graphs in batch
     *
     * Aggregates all node features per graph to get graph-level representation
     *
     * @param node_features [total_nodes, D]
     * @param batch_indices [total_nodes] - which graph each node belongs to
     * @param num_graphs Number of graphs in batch
     * @param pooling Pooling type ("mean", "sum", "max")
     * @return Graph-level features [num_graphs, D]
     */
    static TorchTensor global_pool(
        const TorchTensor& node_features,
        const TorchTensor& batch_indices,
        int num_graphs,
        const std::string& pooling = "mean"
    );
};

/**
 * @brief Multi-hop message passing
 */
class MultiHopAggregation {
public:
    /**
     * @brief K-hop neighborhood aggregation
     *
     * Aggregates messages from k-hop neighbors with learnable weights
     *
     * @param graph Graph structure
     * @param node_features Node features [N, D]
     * @param k Number of hops
     * @param hop_weights Weights for each hop [k]
     * @return Multi-hop aggregated features [N, D]
     */
    static TorchTensor k_hop_aggregate(
        const SparseGraph& graph,
        const TorchTensor& node_features,
        int k,
        const TorchTensor& hop_weights = {}
    );

    /**
     * @brief JK-Net style aggregation
     *
     * Concatenates or pools features from all layers
     */
    static TorchTensor jumping_knowledge(
        const std::vector<TorchTensor>& layer_outputs,
        const std::string& mode = "concat"  // "concat", "max", "lstm"
    );
};

/**
 * @brief Heterogeneous graph message passing
 *
 * For graphs with multiple node/edge types (e.g., gene-protein-metabolite networks)
 */
class HeteroMessagePassing {
public:
    /**
     * @brief Relation-specific message passing
     *
     * Different message functions for different edge types
     *
     * @param graphs Map of edge_type -> graph
     * @param node_features Map of node_type -> features
     * @param target_node_type Which node type to update
     * @return Updated features for target node type
     */
    static TorchTensor hetero_aggregate(
        const std::unordered_map<std::string, SparseGraph>& graphs,
        const std::unordered_map<std::string, TorchTensor>& node_features,
        const std::string& target_node_type
    );
};

} // namespace message_passing
} // namespace omnigraph

#endif // OMNIGRAPH_MESSAGE_PASSING_HPP
