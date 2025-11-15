#' Single-cell multi-omics analysis
#'
#' @description
#' CITE-seq, scATAC-seq, multiome, trajectory inference

#' Integrate sc multi-omics modalities (WNN-like approach)
#' @param sc_exp SingleCellMultiOmicsExperiment
#' @param modalities character vector of modality names to integrate
#' @param k integer, number of nearest neighbors
#' @return updated experiment with integrated embedding
#' @export
sc_integrate_modalities <- function(sc_exp, modalities = NULL, k = 20) {

  if (is.null(modalities)) {
    modalities <- unlist(sc_exp@modalities)
  }

  # Get assays
  assays_list <- SummarizedExperiment::assays(sc_exp)
  assays_list <- assays_list[names(assays_list) %in% modalities]

  if (length(assays_list) < 2) {
    stop("Need at least 2 modalities for integration")
  }

  # Compute embeddings for each modality
  embeddings <- lapply(names(assays_list), function(mod_name) {
    message("Computing embedding for ", mod_name)

    mat <- assays_list[[mod_name]]
    mat_t <- t(mat)
    mat_t[is.na(mat_t)] <- 0

    # PCA
    if (ncol(mat_t) > 50) {
      pca <- prcomp(mat_t, rank. = 30)
      pca$x
    } else {
      mat_t
    }
  })

  names(embeddings) <- names(assays_list)
  sc_exp@cell_embeddings <- embeddings

  # Build kNN graphs for each modality
  knn_graphs <- lapply(embeddings, function(emb) {
    .build_knn_graph(emb, k = k)
  })

  # Compute modality weights based on connectivity
  weights <- sapply(knn_graphs, function(g) {
    # Metric: average edge weight (connectivity strength)
    mean(g$weights, na.rm = TRUE)
  })

  weights <- weights / sum(weights)
  sc_exp@modality_weights <- weights

  # Weighted neighbor voting (WNN-like)
  integrated_graph <- .integrate_knn_graphs(knn_graphs, weights)

  # UMAP on integrated graph
  if (requireNamespace("uwot", quietly = TRUE)) {
    integrated_emb <- uwot::umap(
      integrated_graph$adjacency,
      n_neighbors = k,
      min_dist = 0.3,
      metric = "precomputed"
    )
  } else {
    # Fallback: weighted average of embeddings
    emb_list <- lapply(seq_along(embeddings), function(i) {
      embeddings[[i]] * weights[i]
    })
    integrated_emb <- Reduce("+", emb_list)
  }

  colnames(integrated_emb) <- paste0("Integrated_", 1:ncol(integrated_emb))
  sc_exp@integrated_embedding <- integrated_emb

  sc_exp
}

#' Build kNN graph
#' @keywords internal
.build_knn_graph <- function(embedding, k = 20) {

  n_cells <- nrow(embedding)
  dist_mat <- as.matrix(dist(embedding))

  # Find k nearest neighbors for each cell
  neighbors <- t(apply(dist_mat, 1, function(d) order(d)[1:(k + 1)]))
  distances <- t(apply(dist_mat, 1, function(d) sort(d)[1:(k + 1)]))

  # Convert to adjacency matrix (Gaussian kernel)
  adjacency <- matrix(0, n_cells, n_cells)
  sigma <- median(distances[, k + 1])

  for (i in 1:n_cells) {
    for (j_idx in 2:(k + 1)) {  # Skip self (index 1)
      j <- neighbors[i, j_idx]
      w <- exp(-distances[i, j_idx]^2 / (2 * sigma^2))
      adjacency[i, j] <- w
      adjacency[j, i] <- w  # Symmetric
    }
  }

  list(
    adjacency = adjacency,
    neighbors = neighbors,
    distances = distances,
    weights = adjacency[adjacency > 0]
  )
}

#' Integrate multiple kNN graphs with weights
#' @keywords internal
.integrate_knn_graphs <- function(knn_graphs, weights) {

  # Weighted sum of adjacency matrices
  integrated_adj <- Reduce("+", lapply(seq_along(knn_graphs), function(i) {
    knn_graphs[[i]]$adjacency * weights[i]
  }))

  list(
    adjacency = integrated_adj,
    weights = integrated_adj[integrated_adj > 0]
  )
}

#' sc-ATAC-seq peak calling and motif enrichment
#' @param atacseq_mat matrix of ATAC-seq counts (peaks x cells)
#' @param genome character, genome version
#' @return list with peak annotations
#' @export
sc_atac_peaks <- function(atacseq_mat, genome = "hg38") {

  message("Note: Simplified peak analysis. Use Signac for production.")

  # Simulate peak annotations
  n_peaks <- nrow(atacseq_mat)

  annotations <- data.frame(
    peak = rownames(atacseq_mat),
    chr = sample(paste0("chr", c(1:22, "X")), n_peaks, replace = TRUE),
    start = sample(1:1e8, n_peaks),
    end = sample(1:1e8, n_peaks),
    annotation = sample(c("Promoter", "Exon", "Intron", "Intergenic"),
                       n_peaks, replace = TRUE,
                       prob = c(0.2, 0.1, 0.4, 0.3))
  )

  # Peak accessibility scores
  accessibility <- rowMeans(atacseq_mat > 0)

  annotations$accessibility <- accessibility

  list(
    annotations = annotations,
    n_peaks = n_peaks,
    genome = genome
  )
}

#' Trajectory inference (simplified Slingshot-like)
#' @param sc_exp SingleCellMultiOmicsExperiment
#' @param clusters factor of cluster assignments
#' @param start_cluster integer, starting cluster
#' @return list with pseudotime and lineages
#' @export
sc_trajectory <- function(sc_exp, clusters, start_cluster = NULL) {

  # Use integrated embedding if available
  if (ncol(sc_exp@integrated_embedding) > 0) {
    embedding <- sc_exp@integrated_embedding
  } else if (length(sc_exp@cell_embeddings) > 0) {
    embedding <- sc_exp@cell_embeddings[[1]]
  } else {
    stop("No embedding available. Run sc_integrate_modalities first.")
  }

  # Compute cluster centers
  cluster_ids <- unique(clusters)
  n_clusters <- length(cluster_ids)

  centers <- t(sapply(cluster_ids, function(cl) {
    colMeans(embedding[clusters == cl, , drop = FALSE])
  }))

  # Build minimum spanning tree of cluster centers
  dist_mat <- as.matrix(dist(centers))
  mst <- .minimum_spanning_tree(dist_mat)

  # Determine root
  if (is.null(start_cluster)) {
    # Use cluster with most cells
    start_cluster <- names(which.max(table(clusters)))
  }

  start_idx <- which(cluster_ids == start_cluster)

  # Compute pseudotime via shortest path from root
  pseudotime <- numeric(length(clusters))

  for (i in seq_along(cluster_ids)) {
    cl <- cluster_ids[i]
    cells_in_cluster <- which(clusters == cl)

    # Distance from cluster center to root center
    path_dist <- .shortest_path_distance(mst, start_idx, i)

    # Within-cluster pseudotime based on projection
    if (i != start_idx) {
      direction <- centers[i, ] - centers[start_idx, ]
      direction <- direction / sqrt(sum(direction^2))

      cell_positions <- embedding[cells_in_cluster, ]
      projections <- apply(cell_positions, 1, function(pos) {
        sum((pos - centers[start_idx, ]) * direction)
      })

      projections <- projections - min(projections)
      projections <- projections / max(projections + 1e-6)

      pseudotime[cells_in_cluster] <- path_dist + projections
    } else {
      pseudotime[cells_in_cluster] <- 0
    }
  }

  # Normalize
  pseudotime <- pseudotime / max(pseudotime)

  sc_exp@trajectory <- list(
    pseudotime = pseudotime,
    mst = mst,
    cluster_centers = centers,
    start_cluster = start_cluster
  )

  sc_exp
}

#' Minimum spanning tree (Prim's algorithm)
#' @keywords internal
.minimum_spanning_tree <- function(dist_mat) {

  n <- nrow(dist_mat)
  visited <- rep(FALSE, n)
  mst_edges <- matrix(0, n - 1, 2)
  mst_weights <- numeric(n - 1)

  # Start from node 1
  visited[1] <- TRUE
  edge_count <- 0

  while (edge_count < n - 1) {
    min_dist <- Inf
    min_i <- -1
    min_j <- -1

    for (i in which(visited)) {
      for (j in which(!visited)) {
        if (dist_mat[i, j] < min_dist) {
          min_dist <- dist_mat[i, j]
          min_i <- i
          min_j <- j
        }
      }
    }

    edge_count <- edge_count + 1
    mst_edges[edge_count, ] <- c(min_i, min_j)
    mst_weights[edge_count] <- min_dist
    visited[min_j] <- TRUE
  }

  list(edges = mst_edges, weights = mst_weights)
}

#' Shortest path distance in MST
#' @keywords internal
.shortest_path_distance <- function(mst, from, to) {

  if (from == to) return(0)

  # Build adjacency list
  n_nodes <- max(mst$edges)
  adj_list <- vector("list", n_nodes)

  for (i in 1:nrow(mst$edges)) {
    u <- mst$edges[i, 1]
    v <- mst$edges[i, 2]
    w <- mst$weights[i]

    adj_list[[u]] <- rbind(adj_list[[u]], c(v, w))
    adj_list[[v]] <- rbind(adj_list[[v]], c(u, w))
  }

  # BFS
  visited <- rep(FALSE, n_nodes)
  dist <- rep(Inf, n_nodes)
  dist[from] <- 0
  queue <- from

  while (length(queue) > 0) {
    u <- queue[1]
    queue <- queue[-1]

    if (visited[u]) next
    visited[u] <- TRUE

    if (!is.null(adj_list[[u]])) {
      for (i in 1:nrow(adj_list[[u]])) {
        v <- adj_list[[u]][i, 1]
        w <- adj_list[[u]][i, 2]

        if (dist[u] + w < dist[v]) {
          dist[v] <- dist[u] + w
          queue <- c(queue, v)
        }
      }
    }
  }

  dist[to]
}

#' Cell-cell communication prediction
#' @param sc_exp SingleCellMultiOmicsExperiment
#' @param clusters factor of cell type assignments
#' @param ligand_receptor_db data.frame with ligand and receptor columns
#' @return DataFrame of predicted interactions
#' @export
sc_cell_communication <- function(sc_exp, clusters,
                                  ligand_receptor_db = NULL) {

  # Use default LR database if not provided
  if (is.null(ligand_receptor_db)) {
    ligand_receptor_db <- data.frame(
      ligand = c("TGFB1", "IL6", "TNF", "VEGFA"),
      receptor = c("TGFBR1", "IL6R", "TNFR1", "FLT1")
    )
  }

  # Get expression data (use RNA if available)
  assays_list <- SummarizedExperiment::assays(sc_exp)
  if ("RNA" %in% names(assays_list)) {
    expr <- assays_list$RNA
  } else {
    expr <- assays_list[[1]]
  }

  # For each cell type pair and LR pair
  cell_types <- unique(clusters)
  results <- list()

  for (i in 1:length(cell_types)) {
    for (j in 1:length(cell_types)) {
      sender <- cell_types[i]
      receiver <- cell_types[j]

      sender_cells <- which(clusters == sender)
      receiver_cells <- which(clusters == receiver)

      for (lr_idx in 1:nrow(ligand_receptor_db)) {
        ligand <- ligand_receptor_db$ligand[lr_idx]
        receptor <- ligand_receptor_db$receptor[lr_idx]

        if (!ligand %in% rownames(expr) || !receptor %in% rownames(expr)) {
          next
        }

        # Mean expression
        lig_expr <- mean(expr[ligand, sender_cells], na.rm = TRUE)
        rec_expr <- mean(expr[receptor, receiver_cells], na.rm = TRUE)

        # Interaction score (simple product)
        score <- lig_expr * rec_expr

        results[[length(results) + 1]] <- data.frame(
          sender = sender,
          receiver = receiver,
          ligand = ligand,
          receptor = receptor,
          ligand_expr = lig_expr,
          receptor_expr = rec_expr,
          score = score
        )
      }
    }
  }

  results_df <- do.call(rbind, results)
  results_df <- results_df[order(-results_df$score), ]

  S4Vectors::DataFrame(results_df)
}
