#' @import methods SummarizedExperiment S4Vectors MultiAssayExperiment BiocParallel ggplot2 stats utils
NULL

#' Get default parallel backend
#' @keywords internal
.omni_bpparam <- function() {
  tryCatch(BiocParallel::bpparam(), error = function(e) BiocParallel::SerialParam())
}

