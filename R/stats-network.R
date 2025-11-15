#' Network analysis for omics data
#'
#' @description
#' WGCNA, gene regulatory networks, protein-protein interactions, metabolic networks

#' WGCNA co-expression network analysis
#' @param oe OmicsExperiment
#' @param power integer, soft thresholding power
#' @param min_module_size integer, minimum module size
#' @param merge_cut_height numeric, height to merge similar modules
#' @param trait optional trait data for module-trait associations
#' @return list with modules and network
#' @export
network_wgcna <- function(oe, power = NULL, min_module_size = 30,
                          merge_cut_height = 0.25, trait = NULL) {

  if (!requireNamespace("WGCNA", quietly = TRUE)) {
    warning("WGCNA not available, using simple correlation clustering")
    return(.simple_coexpression(oe, min_module_size))
  }

  # Extract data (samples x features)
  dat_expr <- t(SummarizedExperiment::assay(oe, 1))
  dat_expr[is.na(dat_expr)] <- 0

  # Check for good genes/samples
  gsg <- WGCNA::goodSamplesGenes(dat_expr, verbose = 0)
  if (!gsg$allOK) {
    dat_expr <- dat_expr[gsg$goodSamples, gsg$goodGenes]
  }

  # Choose soft threshold if not provided
  if (is.null(power)) {
    powers <- c(seq(1, 10, by = 1), seq(12, 20, by = 2))
    sft <- WGCNA::pickSoftThreshold(dat_expr, powerVector = powers, verbose = 0)
    power <- sft$powerEstimate
    if (is.na(power)) power <- 6
  }

  # Calculate adjacency
  adjacency <- WGCNA::adjacency(dat_expr, power = power)

  # TOM (Topological Overlap Matrix)
  TOM <- WGCNA::TOMsimilarity(adjacency)
  dissTOM <- 1 - TOM

  # Hierarchical clustering
  gene_tree <- hclust(as.dist(dissTOM), method = "average")

  # Module detection
  dynamic_mods <- WGCNA::cutreeDynamic(
    dendro = gene_tree,
    distM = dissTOM,
    deepSplit = 2,
    pamRespectsDendro = FALSE,
    minClusterSize = min_module_size
  )

  # Merge similar modules
  ME_list <- WGCNA::moduleEigengenes(dat_expr, colors = dynamic_mods)
  MEs <- ME_list$eigengenes
  ME_diss <- 1 - cor(MEs)
  ME_tree <- hclust(as.dist(ME_diss), method = "average")

  merged_colors <- WGCNA::mergeCloseModules(
    dat_expr,
    dynamic_mods,
    cutHeight = merge_cut_height,
    verbose = 0
  )

  modules <- merged_colors$colors
  names(modules) <- colnames(dat_expr)

  # Module eigengenes
  MEs_final <- merged_colors$newMEs

  # Module-trait correlation if trait provided
  module_trait_cor <- NULL
  if (!is.null(trait)) {
    if (is.vector(trait)) trait <- matrix(trait, ncol = 1)
    module_trait_cor <- cor(MEs_final, trait, use = "p")
    module_trait_p <- WGCNA::corPvalueStudent(module_trait_cor, nrow(dat_expr))
  }

  # Hub genes
  hub_genes <- .get_hub_genes(dat_expr, modules, MEs_final)

  list(
    modules = modules,
    module_eigengenes = MEs_final,
    TOM = TOM,
    gene_tree = gene_tree,
    power = power,
    module_trait_cor = module_trait_cor,
    hub_genes = hub_genes,
    n_modules = length(unique(modules))
  )
}

#' Get hub genes from WGCNA
#' @keywords internal
.get_hub_genes <- function(dat_expr, modules, MEs, top_n = 10) {

  module_names <- unique(modules)
  module_names <- module_names[module_names != "0"]  # Exclude grey module

  hubs <- lapply(module_names, function(mod) {
    in_module <- modules == mod
    module_data <- dat_expr[, in_module]

    # Module membership
    MM <- cor(module_data, MEs[, paste0("ME", mod)], use = "p")

    # Intramodular connectivity
    kWithin <- apply(abs(cor(module_data)), 2, sum) - 1

    hub_df <- data.frame(
      gene = colnames(module_data),
      module = mod,
      MM = MM[, 1],
      kWithin = kWithin
    )

    hub_df <- hub_df[order(-hub_df$kWithin), ]
    head(hub_df, top_n)
  })

  do.call(rbind, hubs)
}

#' Simple co-expression clustering fallback
#' @keywords internal
.simple_coexpression <- function(oe, min_module_size) {

  dat_expr <- t(SummarizedExperiment::assay(oe, 1))
  dat_expr[is.na(dat_expr)] <- 0

  # Correlation
  cor_mat <- cor(dat_expr)
  dist_mat <- as.dist(1 - abs(cor_mat))

  # Hierarchical clustering
  hc <- hclust(dist_mat, method = "average")
  modules <- cutree(hc, h = 0.5)

  # Filter small modules
  module_sizes <- table(modules)
  keep_modules <- names(module_sizes)[module_sizes >= min_module_size]
  modules[!modules %in% keep_modules] <- 0

  list(
    modules = modules,
    module_eigengenes = NULL,
    TOM = NULL,
    gene_tree = hc,
    power = NULL,
    n_modules = length(unique(modules)) - 1,
    method = "simple_correlation_clustering"
  )
}

#' Gene regulatory network inference
#' @param oe OmicsExperiment
#' @param method character, method for GRN inference
#' @param regulators character vector of regulator gene names
#' @return list with GRN
#' @export
network_grn <- function(oe, method = "genie3", regulators = NULL) {

  if (!requireNamespace("GENIE3", quietly = TRUE)) {
    warning("GENIE3 not available, using correlation-based GRN")
    return(.correlation_grn(oe, regulators))
  }

  # Extract expression
  expr_mat <- SummarizedExperiment::assay(oe, 1)
  expr_mat[is.na(expr_mat)] <- 0

  # Select top variable genes
  vars <- apply(expr_mat, 1, var)
  top_idx <- order(-vars)[1:min(1000, nrow(expr_mat))]
  expr_subset <- expr_mat[top_idx, ]

  # Run GENIE3
  if (is.null(regulators)) {
    regulators <- 1:nrow(expr_subset)
  } else {
    regulators <- which(rownames(expr_subset) %in% regulators)
  }

  weight_matrix <- GENIE3::GENIE3(
    expr_subset,
    regulators = regulators,
    nCores = 1
  )

  # Get link list
  link_list <- GENIE3::getLinkList(weight_matrix, threshold = 0.01)

  # Rename
  link_list$regulatoryGene <- rownames(expr_subset)[link_list$regulatoryGene]
  link_list$targetGene <- rownames(expr_subset)[link_list$targetGene]

  list(
    weight_matrix = weight_matrix,
    edges = link_list,
    n_edges = nrow(link_list),
    method = "GENIE3"
  )
}

#' Correlation-based GRN fallback
#' @keywords internal
.correlation_grn <- function(oe, regulators = NULL, threshold = 0.5) {

  expr_mat <- SummarizedExperiment::assay(oe, 1)
  expr_mat[is.na(expr_mat)] <- 0

  # Select top variable genes
  vars <- apply(expr_mat, 1, var)
  top_idx <- order(-vars)[1:min(200, nrow(expr_mat))]
  expr_subset <- expr_mat[top_idx, ]

  # Correlation
  cor_mat <- cor(t(expr_subset))

  # Extract edges above threshold
  edges_idx <- which(abs(cor_mat) > threshold & cor_mat != 1, arr.ind = TRUE)
  edges_idx <- edges_idx[edges_idx[, 1] < edges_idx[, 2], ]

  if (nrow(edges_idx) == 0) {
    edges <- data.frame(
      regulatoryGene = character(0),
      targetGene = character(0),
      weight = numeric(0)
    )
  } else {
    edges <- data.frame(
      regulatoryGene = rownames(expr_subset)[edges_idx[, 1]],
      targetGene = rownames(expr_subset)[edges_idx[, 2]],
      weight = cor_mat[edges_idx]
    )
  }

  list(
    weight_matrix = cor_mat,
    edges = edges,
    n_edges = nrow(edges),
    method = "correlation_fallback"
  )
}

#' Protein-protein interaction network enrichment
#' @param gene_list character vector of gene symbols
#' @param ppi_db character, PPI database to use
#' @param species character, species name
#' @return list with PPI subnetwork
#' @export
network_ppi <- function(gene_list, ppi_db = "string", species = "human") {

  # Simulate PPI network (in real implementation, would query STRING DB)
  message("Note: This is a simulated PPI network. In production, use STRINGdb package.")

  n_genes <- length(gene_list)
  n_edges <- min(n_genes * 5, n_genes * (n_genes - 1) / 2)

  edges <- data.frame(
    gene_a = sample(gene_list, n_edges, replace = TRUE),
    gene_b = sample(gene_list, n_edges, replace = TRUE),
    confidence = runif(n_edges, 0.4, 1.0),
    interaction_type = sample(c("binding", "activation", "inhibition"), n_edges, replace = TRUE)
  )

  # Remove self-loops
  edges <- edges[edges$gene_a != edges$gene_b, ]

  list(
    edges = edges,
    n_nodes = n_genes,
    n_edges = nrow(edges),
    ppi_db = ppi_db,
    species = species
  )
}

#' Network visualization
#' @param network_result result from network_wgcna or network_grn
#' @param layout character, layout algorithm
#' @param top_edges integer, number of top edges to plot
#' @return igraph plot
#' @export
plot_network <- function(network_result, layout = "fr", top_edges = 100) {

  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop("igraph package required for network visualization")
  }

  # Extract edges
  if ("edges" %in% names(network_result)) {
    edges <- network_result$edges
    if (nrow(edges) > top_edges) {
      edges <- head(edges[order(-abs(edges[, 3])), ], top_edges)
    }

    # Create graph
    if (ncol(edges) >= 3) {
      g <- igraph::graph_from_data_frame(edges[, 1:2], directed = TRUE)
      igraph::E(g)$weight <- abs(edges[, 3])
    } else {
      g <- igraph::graph_from_data_frame(edges, directed = TRUE)
    }

  } else if ("modules" %in% names(network_result)) {
    # WGCNA: create module graph
    modules <- network_result$modules
    module_names <- unique(modules)
    module_sizes <- table(modules)

    # Create simple graph of modules
    nodes <- data.frame(
      module = names(module_sizes),
      size = as.numeric(module_sizes)
    )

    g <- igraph::make_empty_graph(n = nrow(nodes), directed = FALSE)
    igraph::V(g)$name <- nodes$module
    igraph::V(g)$size <- nodes$size / max(nodes$size) * 20

  } else {
    stop("Unrecognized network result format")
  }

  # Layout
  if (layout == "fr") {
    lo <- igraph::layout_with_fr(g)
  } else if (layout == "kk") {
    lo <- igraph::layout_with_kk(g)
  } else {
    lo <- igraph::layout_nicely(g)
  }

  # Plot
  plot(g, layout = lo, vertex.label.cex = 0.7, edge.arrow.size = 0.3)

  invisible(g)
}
