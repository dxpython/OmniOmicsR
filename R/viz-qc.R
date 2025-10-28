#' QC plot: total counts vs missing fraction
#' @param object OmicsExperiment
#' @export
plot_qc <- function(object) {
  stopifnot(is(object, "OmicsExperiment"))
  cd <- as.data.frame(SummarizedExperiment::colData(object))
  if (!all(c("total_counts", "missing_frac") %in% colnames(cd))) {
    object <- qc_basic(object)
    cd <- as.data.frame(SummarizedExperiment::colData(object))
  }
  ggplot2::ggplot(cd, ggplot2::aes(x = total_counts, y = missing_frac)) +
    ggplot2::geom_point() + ggplot2::theme_minimal() +
    ggplot2::labs(x = "Total Counts", y = "Missing Fraction", title = "QC: Library size vs Missingness")
}

#' PCA plot on selected assay
#' @param object OmicsExperiment
#' @param assay assay index or name
#' @export
plot_pca <- function(object, assay = 1) {
  stopifnot(is(object, "OmicsExperiment"))
  x <- SummarizedExperiment::assay(object, assay)
  x <- t(scale(t(as.matrix(x)), center = TRUE, scale = TRUE))
  p <- stats::prcomp(t(x))
  df <- data.frame(PC1 = p$x[, 1], PC2 = p$x[, 2], sample = rownames(p$x))
  ggplot2::ggplot(df, ggplot2::aes(PC1, PC2, label = sample)) +
    ggplot2::geom_point() + ggplot2::theme_minimal() + ggplot2::labs(title = "PCA")
}

#' UMAP on selected assay
#' @param object OmicsExperiment
#' @param assay assay name or index
#' @param n_neighbors, min_dist, n_components parameters for UMAP
#' @export
plot_umap <- function(object, assay = 1, n_neighbors = 15, min_dist = 0.1, n_components = 2) {
  if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot not installed")
  x <- SummarizedExperiment::assay(object, assay)
  x <- t(scale(t(as.matrix(x)), center = TRUE, scale = TRUE))
  emb <- uwot::umap(t(x), n_neighbors = n_neighbors, min_dist = min_dist, n_components = n_components, verbose = FALSE)
  df <- as.data.frame(emb)
  names(df) <- paste0("UMAP", seq_len(n_components))
  if (n_components >= 2) {
    p <- ggplot2::ggplot(df, ggplot2::aes_string(x = names(df)[1], y = names(df)[2])) +
      ggplot2::geom_point() + theme_omni() + ggplot2::labs(title = "UMAP")
    return(p)
  } else {
    return(df)
  }
}
