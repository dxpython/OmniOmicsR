#' Select highly variable genes (single-cell)
#'
#' Uses scran::modelGeneVar if available; otherwise ranks by variance on the given assay.
#' @param object OmicsExperiment
#' @param assay assay name or index
#' @param top number of HVGs to return
#' @return character vector of HVG feature IDs
#' @export
sc_hvg <- function(object, assay = 1, top = 2000) {
  x <- SummarizedExperiment::assay(object, assay)
  if (requireNamespace("scran", quietly = TRUE)) {
    v <- scran::modelGeneVar(x)
    ord <- order(v$bio, decreasing = TRUE)
    feats <- rownames(x)[ord[seq_len(min(top, nrow(x)))]]
    return(feats)
  }
  vars <- apply(x, 1, stats::var, na.rm = TRUE)
  vars[is.na(vars)] <- 0
  names(sort(vars, decreasing = TRUE))[seq_len(min(top, length(vars)))]
}

