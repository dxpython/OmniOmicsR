#' Basic QC metrics
#'
#' Adds simple QC columns to colData: total_counts, missing_frac
#' @param object OmicsExperiment
#' @return OmicsExperiment
#' @export
qc_basic <- function(object) {
  stopifnot(is(object, "OmicsExperiment"))
  x <- SummarizedExperiment::assay(object, 1)
  total_counts <- colSums(x, na.rm = TRUE)
  missing_frac <- colMeans(is.na(x))
  cd <- SummarizedExperiment::colData(object)
  cd$total_counts <- total_counts[colnames(x)]
  cd$missing_frac <- missing_frac[colnames(x)]
  SummarizedExperiment::colData(object) <- cd
  object <- .add_log(object, "qc_basic")
  object
}

#' Mitochondrial gene ratio per sample
#'
#' For human default pattern '^MT-' (case-insensitive), mouse '^mt-'.
#' @param object OmicsExperiment
#' @param pattern regex to identify mitochondrial features (matched on rownames)
#' @export
qc_mito_ratio <- function(object, pattern = "^(?i)MT-") {
  stopifnot(is(object, "OmicsExperiment"))
  x <- SummarizedExperiment::assay(object, 1)
  feats <- rownames(object)
  mito <- grepl(pattern, feats, perl = TRUE)
  mito_counts <- colSums(x[mito, , drop = FALSE], na.rm = TRUE)
  total_counts <- colSums(x, na.rm = TRUE)
  ratio <- ifelse(total_counts > 0, mito_counts / pmax(total_counts, 1), NA_real_)
  cd <- SummarizedExperiment::colData(object)
  cd$mito_ratio <- ratio[colnames(x)]
  SummarizedExperiment::colData(object) <- cd
  .add_log(object, "qc_mito_ratio", list(pattern = pattern))
}

#' Simple saturation proxy: detected features vs library size
#'
#' Adds 'detected_features' to colData and returns object.
#' @param object OmicsExperiment
#' @param threshold value above which a feature is considered detected (default >0)
#' @export
qc_saturation <- function(object, threshold = 0) {
  stopifnot(is(object, "OmicsExperiment"))
  x <- SummarizedExperiment::assay(object, 1)
  detected <- colSums(x > threshold, na.rm = TRUE)
  cd <- SummarizedExperiment::colData(object)
  cd$detected_features <- detected[colnames(x)]
  SummarizedExperiment::colData(object) <- cd
  .add_log(object, "qc_saturation", list(threshold = threshold))
}
