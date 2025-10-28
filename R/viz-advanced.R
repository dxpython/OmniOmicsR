#' Omni theme for ggplot2
#' @param base_size base font size
#' @export
theme_omni <- function(base_size = 11) {
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = "bold"),
      legend.position = "right"
    )
}

#' Volcano plot from DE results
#' @param res data.frame with columns log2FoldChange (or logFC), pvalue or padj
#' @param lfc_col column name for log2FC
#' @param p_col column name for p-value (prefers padj)
#' @param alpha FDR threshold
#' @param top_n label top n significant features
#' @export
plot_volcano <- function(res, lfc_col = NULL, p_col = NULL, alpha = 0.05, top_n = 10) {
  if (is.null(lfc_col)) lfc_col <- if ("log2FoldChange" %in% names(res)) "log2FoldChange" else if ("logFC" %in% names(res)) "logFC" else stop("Provide lfc_col")
  if (is.null(p_col)) p_col <- if ("padj" %in% names(res)) "padj" else if ("PValue" %in% names(res)) "PValue" else if ("pvalue" %in% names(res)) "pvalue" else stop("Provide p_col")
  df <- res
  df$neglog10p <- -log10(pmax(df[[p_col]], .Machine$double.xmin))
  df$sig <- !is.na(df[[p_col]]) & df[[p_col]] <= alpha
  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data[[lfc_col]], y = neglog10p, color = sig)) +
    ggplot2::geom_point(alpha = 0.6) +
    ggplot2::labs(x = "log2 Fold Change", y = "-log10(p)", title = "Volcano Plot") +
    theme_omni()
  if (top_n > 0 && ("feature" %in% names(df) || !is.null(rownames(df)))) {
    df$feature_label <- if ("feature" %in% names(df)) df$feature else rownames(df)
    top_idx <- order(df[[p_col]])[seq_len(min(top_n, sum(df$sig, na.rm = TRUE)))]
    lab <- df[top_idx, , drop = FALSE]
    p <- p + ggplot2::geom_text(data = lab, ggplot2::aes(label = feature_label), vjust = -0.5, size = 3, show.legend = FALSE)
  }
  p
}

.row_z <- function(mat) {
  if (exists("fast_row_z")) return(fast_row_z(mat))
  m <- rowMeans(mat, na.rm = TRUE)
  s <- apply(mat, 1, stats::sd, na.rm = TRUE); s[s == 0] <- 1
  sweep(sweep(mat, 1, m, "-"), 1, s, "/")
}

#' Heatmap for an assay
#' @param object OmicsExperiment
#' @param assay assay name or index
#' @param features character vector of feature IDs; if NULL, use top variable
#' @param top number of top variable features if features is NULL
#' @param scale scale rows ("row") or none
#' @export
plot_heatmap <- function(object, assay = "norm", features = NULL, top = 50, scale = c("row","none")) {
  scale <- match.arg(scale)
  x <- SummarizedExperiment::assay(object, assay)
  if (is.null(features)) {
    vars <- apply(x, 1, stats::var, na.rm = TRUE)
    vars[is.na(vars)] <- 0
    features <- names(sort(vars, decreasing = TRUE))[seq_len(min(top, nrow(x)))]
  }
  x <- x[features, , drop = FALSE]
  if (scale == "row") x <- .row_z(x)
  df <- data.frame(feature = rep(rownames(x), times = ncol(x)),
                   sample = rep(colnames(x), each = nrow(x)),
                   value = as.vector(x))
  ggplot2::ggplot(df, ggplot2::aes(sample, feature, fill = value)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_gradient2(low = "#2166ac", mid = "#f7f7f7", high = "#b2182b", midpoint = 0) +
    ggplot2::labs(x = NULL, y = NULL, title = "Heatmap") +
    theme_omni() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
}

#' Batch effect before/after comparison via PCA
#' @param object OmicsExperiment
#' @param assay_before assay for before
#' @param assay_after assay for after (e.g., 'combat')
#' @export
plot_batch_compare <- function(object, assay_before = 1, assay_after = "combat") {
  xb <- SummarizedExperiment::assay(object, assay_before)
  xa <- SummarizedExperiment::assay(object, assay_after)
  xb <- t(scale(t(xb)))
  xa <- t(scale(t(xa)))
  pb <- stats::prcomp(t(xb))$x[, 1:2, drop = FALSE]
  pa <- stats::prcomp(t(xa))$x[, 1:2, drop = FALSE]
  dfb <- data.frame(PC1 = pb[,1], PC2 = pb[,2], sample = rownames(pb), state = "before")
  dfa <- data.frame(PC1 = pa[,1], PC2 = pa[,2], sample = rownames(pa), state = "after")
  d <- rbind(dfb, dfa)
  ggplot2::ggplot(d, ggplot2::aes(PC1, PC2)) +
    ggplot2::geom_point(alpha = 0.7) +
    ggplot2::facet_wrap(~state) +
    ggplot2::labs(title = "Batch correction comparison") +
    theme_omni()
}
