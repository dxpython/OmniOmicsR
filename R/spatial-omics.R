#' Spatial omics analysis
#'
#' @description
#' Spatial transcriptomics, proteomics, and metabolomics analysis

#' Spatial variable features detection
#' @param spe SpatialOmicsExperiment
#' @param method character, method for spatial variability
#' @param n_top integer, number of top features
#' @return vector of spatially variable features
#' @export
spatial_variable_features <- function(spe, method = "moranI", n_top = 1000) {

  expr <- SummarizedExperiment::assay(spe, 1)
  coords <- spe@spatial_coords

  if (method == "moranI") {
    # Moran's I for spatial autocorrelation
    scores <- apply(expr, 1, function(gene_expr) {
      .moran_i(gene_expr, coords)
    })

  } else if (method == "gearyC") {
    # Geary's C
    scores <- apply(expr, 1, function(gene_expr) {
      .geary_c(gene_expr, coords)
    })

  } else {
    # Simple variance-based
    scores <- apply(expr, 1, var)
  }

  # Get top features
  top_idx <- order(-scores)[1:min(n_top, length(scores))]
  top_features <- rownames(expr)[top_idx]

  data.frame(
    feature = top_features,
    score = scores[top_idx]
  )
}

#' Moran's I statistic
#' @keywords internal
.moran_i <- function(x, coords) {

  n <- length(x)
  if (n != nrow(coords)) stop("Mismatch in x and coords")

  # Distance matrix
  dist_mat <- as.matrix(dist(coords))

  # Spatial weights (inverse distance)
  w <- 1 / (dist_mat + diag(n))
  diag(w) <- 0
  w <- w / sum(w)

  # Moran's I
  x_centered <- x - mean(x)
  numerator <- sum(w * outer(x_centered, x_centered, "*"))
  denominator <- sum(x_centered^2) / n

  I <- numerator / denominator
  return(I)
}

#' Geary's C statistic
#' @keywords internal
.geary_c <- function(x, coords) {

  n <- length(x)
  dist_mat <- as.matrix(dist(coords))

  w <- 1 / (dist_mat + diag(n))
  diag(w) <- 0
  w <- w / sum(w)

  # Geary's C
  numerator <- sum(w * outer(x, x, function(a, b) (a - b)^2))
  denominator <- 2 * sum(w) * var(x)

  C <- numerator / denominator
  return(-C)  # Negative so higher is more spatially variable
}

#' Spatial clustering
#' @param spe SpatialOmicsExperiment
#' @param n_clusters integer, number of spatial domains
#' @param features character vector of features to use
#' @param method character, clustering method
#' @return factor of cluster assignments
#' @export
spatial_clustering <- function(spe, n_clusters = 5, features = NULL, method = "leiden") {

  expr <- SummarizedExperiment::assay(spe, 1)
  coords <- spe@spatial_coords

  # Select features
  if (is.null(features)) {
    # Use top variable features
    vars <- apply(expr, 1, var)
    top_idx <- order(-vars)[1:min(2000, nrow(expr))]
    expr <- expr[top_idx, ]
  } else {
    expr <- expr[features, ]
  }

  # Combine expression and spatial info
  expr_t <- t(expr)
  expr_t <- scale(expr_t)
  expr_t[is.na(expr_t)] <- 0

  # Add weighted spatial coordinates
  coords_scaled <- scale(coords)
  spatial_weight <- 0.3
  combined <- cbind(expr_t, coords_scaled * spatial_weight)

  # Clustering
  if (method == "kmeans") {
    clusters <- kmeans(combined, centers = n_clusters, nstart = 20)$cluster

  } else if (method == "leiden" && requireNamespace("igraph", quietly = TRUE)) {
    # Build kNN graph
    knn <- 10
    dist_mat <- as.matrix(dist(combined))
    neighbors <- t(apply(dist_mat, 1, function(d) order(d)[1:(knn + 1)]))

    # Create graph
    edges <- NULL
    for (i in 1:nrow(neighbors)) {
      for (j in neighbors[i, -1]) {
        if (i < j) {
          edges <- rbind(edges, c(i, j))
        }
      }
    }

    g <- igraph::graph_from_edgelist(edges, directed = FALSE)

    # Leiden clustering
    if (requireNamespace("leiden", quietly = TRUE)) {
      adj_matrix <- igraph::as_adjacency_matrix(g, sparse = TRUE)
      clusters <- leiden::leiden(adj_matrix, resolution_parameter = 0.5)
    } else {
      # Fallback to Louvain
      clusters <- igraph::cluster_louvain(g)$membership
    }

  } else {
    # Hierarchical clustering
    hc <- hclust(dist(combined), method = "ward.D2")
    clusters <- cutree(hc, k = n_clusters)
  }

  factor(clusters)
}

#' Spatial statistics (local patterns)
#' @param spe SpatialOmicsExperiment
#' @param feature character, feature name
#' @param radius numeric, neighborhood radius
#' @return vector of local statistics
#' @export
spatial_local_stats <- function(spe, feature, radius = 100) {

  expr <- SummarizedExperiment::assay(spe, 1)[feature, ]
  coords <- spe@spatial_coords

  # For each spot, compute local mean in neighborhood
  n_spots <- nrow(coords)
  local_means <- numeric(n_spots)

  dist_mat <- as.matrix(dist(coords))

  for (i in 1:n_spots) {
    neighbors <- which(dist_mat[i, ] <= radius)
    local_means[i] <- mean(expr[neighbors], na.rm = TRUE)
  }

  data.frame(
    spot = colnames(spe),
    expression = expr,
    local_mean = local_means,
    local_diff = expr - local_means
  )
}

#' Spatial visualization
#' @param spe SpatialOmicsExperiment
#' @param feature character, feature to plot (or "clusters")
#' @param clusters optional cluster assignments
#' @param point_size numeric
#' @return ggplot2 object
#' @export
plot_spatial <- function(spe, feature = NULL, clusters = NULL, point_size = 1.5) {

  coords <- as.data.frame(spe@spatial_coords)
  colnames(coords) <- c("x", "y")[1:ncol(coords)]

  if (!is.null(feature) && feature != "clusters") {
    # Plot feature expression
    expr <- SummarizedExperiment::assay(spe, 1)[feature, ]
    coords$value <- expr
    color_var <- "value"
    fill_label <- feature

  } else if (!is.null(clusters)) {
    # Plot clusters
    coords$cluster <- as.factor(clusters)
    color_var <- "cluster"
    fill_label <- "Cluster"

  } else {
    stop("Must provide either feature or clusters")
  }

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p <- ggplot2::ggplot(coords, ggplot2::aes(x = x, y = y)) +
      ggplot2::geom_point(ggplot2::aes(color = .data[[color_var]]),
                         size = point_size) +
      ggplot2::coord_fixed() +
      ggplot2::theme_minimal() +
      ggplot2::labs(color = fill_label, fill = fill_label)

    if (color_var == "value") {
      p <- p + ggplot2::scale_color_viridis_c()
    } else {
      p <- p + ggplot2::scale_color_discrete()
    }

    return(p)

  } else {
    # Base R plot
    if (color_var == "value") {
      plot(coords$x, coords$y, col = heat.colors(100)[cut(coords$value, 100)],
           pch = 16, cex = point_size,
           xlab = "X", ylab = "Y", main = feature)
    } else {
      plot(coords$x, coords$y, col = coords$cluster,
           pch = 16, cex = point_size,
           xlab = "X", ylab = "Y", main = "Clusters")
    }
  }
}

#' Spatial trajectory inference
#' @param spe SpatialOmicsExperiment
#' @param start_cluster integer, starting cluster/region
#' @param end_cluster integer, ending cluster/region
#' @param clusters cluster assignments
#' @return list with pseudotime and trajectory
#' @export
spatial_trajectory <- function(spe, start_cluster, end_cluster, clusters) {

  coords <- spe@spatial_coords
  expr <- t(SummarizedExperiment::assay(spe, 1))

  # Find representative points for start and end
  start_spots <- which(clusters == start_cluster)
  end_spots <- which(clusters == end_cluster)

  start_center <- colMeans(coords[start_spots, , drop = FALSE])
  end_center <- colMeans(coords[end_spots, , drop = FALSE])

  # Compute pseudotime as projection onto start-end axis
  direction <- end_center - start_center
  direction <- direction / sqrt(sum(direction^2))

  pseudotime <- apply(coords, 1, function(spot) {
    sum((spot - start_center) * direction)
  })

  # Normalize to [0, 1]
  pseudotime <- (pseudotime - min(pseudotime)) / (max(pseudotime) - min(pseudotime))

  list(
    pseudotime = pseudotime,
    start_center = start_center,
    end_center = end_center,
    direction = direction
  )
}

#' Cell-cell communication in spatial context
#' @param spe SpatialOmicsExperiment
#' @param ligand_receptor_pairs data.frame with ligand and receptor columns
#' @param distance_threshold numeric, max distance for interaction
#' @return data.frame of predicted interactions
#' @export
spatial_cell_communication <- function(spe, ligand_receptor_pairs,
                                       distance_threshold = 100) {

  expr <- SummarizedExperiment::assay(spe, 1)
  coords <- spe@spatial_coords

  # Distance matrix
  dist_mat <- as.matrix(dist(coords))

  # For each ligand-receptor pair
  results <- lapply(1:nrow(ligand_receptor_pairs), function(i) {
    ligand <- ligand_receptor_pairs$ligand[i]
    receptor <- ligand_receptor_pairs$receptor[i]

    if (!ligand %in% rownames(expr) || !receptor %in% rownames(expr)) {
      return(NULL)
    }

    lig_expr <- expr[ligand, ]
    rec_expr <- expr[receptor, ]

    # Find potential interactions (nearby spots)
    interactions <- which(dist_mat > 0 & dist_mat <= distance_threshold, arr.ind = TRUE)
    interactions <- interactions[interactions[, 1] < interactions[, 2], ]

    if (nrow(interactions) == 0) return(NULL)

    # Compute interaction strength
    interaction_scores <- lig_expr[interactions[, 1]] * rec_expr[interactions[, 2]]

    data.frame(
      spot_ligand = interactions[, 1],
      spot_receptor = interactions[, 2],
      ligand = ligand,
      receptor = receptor,
      distance = dist_mat[interactions],
      score = interaction_scores
    )
  })

  do.call(rbind, results[!sapply(results, is.null)])
}
