#' Circos-style integration plot (chord diagram)
#'
#' Builds a simple chord diagram of feature-feature associations between two assays using
#' strong absolute correlations above a threshold.
#' @param x OmicsExperiment (block A)
#' @param y OmicsExperiment (block B)
#' @param assay_x,assay_y assay names/indices
#' @param threshold absolute correlation threshold
#' @param top maximum number of edges to draw
#' @export
plot_circos_integrate <- function(x, y, assay_x = 1, assay_y = 1, threshold = 0.7, top = 200) {
  if (!requireNamespace("circlize", quietly = TRUE)) stop("circlize not installed")
  cor_df <- assoc_corr(x, y, assay_x = assay_x, assay_y = assay_y)
  cor_df <- cor_df[is.finite(cor_df$cor), , drop = FALSE]
  cor_df <- cor_df[order(-abs(cor_df$cor)), , drop = FALSE]
  cor_df <- cor_df[abs(cor_df$cor) >= threshold, , drop = FALSE]
  if (nrow(cor_df) == 0) stop("No edges pass threshold")
  if (!is.null(top)) cor_df <- head(cor_df, top)
  # Prepare sectors
  labs_x <- unique(cor_df$feature_x)
  labs_y <- unique(cor_df$feature_y)
  sectors <- c(labs_x, labs_y)
  circlize::circos.clear()
  circlize::circos.par(start.degree = 90, gap.after = c(rep(1, length(labs_x) - 1), 8, rep(1, length(labs_y) - 1), 8))
  circlize::circos.initialize(factors = sectors, x = rep(1, length(sectors)))
  circlize::circos.track(ylim = c(0, 1), panel.fun = function(x, y) {
    sector.name <- circlize::get.cell.meta.data("sector.index")
    circlize::circos.text(0.5, 0.9, sector.name, facing = "inside", cex = 0.5)
  })
  # Links
  for (i in seq_len(nrow(cor_df))) {
    a <- cor_df$feature_x[i]; b <- cor_df$feature_y[i]
    col <- ifelse(cor_df$cor[i] > 0, "#1b9e77", "#d95f02")
    circlize::circos.link(a, 0.5, b, 0.5, col = col, lwd = 1)
  }
  invisible(TRUE)
}

