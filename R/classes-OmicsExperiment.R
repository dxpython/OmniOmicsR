#' OmicsExperiment class
#'
#' Extends SummarizedExperiment with omics_type, processing_log, and feature_map.
#' @slot omics_type character
#' @slot processing_log list
#' @slot feature_map DataFrame
#' @exportClass OmicsExperiment
setClass("OmicsExperiment",
  contains = "SummarizedExperiment",
  slots = c(
    omics_type = "character",
    processing_log = "list",
    feature_map = "DataFrame"
  )
)

#' Show method
#' @param object OmicsExperiment
setMethod("show", "OmicsExperiment", function(object) {
  cat("OmicsExperiment<", object@omics_type, "> with ",
      nrow(object), " features and ", ncol(object), " samples\n", sep = "")
})

#' Construct an OmicsExperiment from a matrix/data.frame
#'
#' @param x numeric matrix/data.frame (features x samples)
#' @param omics_type character like 'rna', 'protein', 'metabolite'
#' @param se_assay_name assay name, default 'counts'
#' @param col_data optional DataFrame for samples
#' @param row_data optional DataFrame for features
#' @return OmicsExperiment
#' @export
as_oe <- function(x, omics_type = "rna", se_assay_name = "counts",
                  col_data = NULL, row_data = NULL) {
  stopifnot(is.matrix(x) || is.data.frame(x))
  if (is.data.frame(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  se <- SummarizedExperiment::SummarizedExperiment(
    assays = setNames(list(x), se_assay_name),
    colData = if (is.null(col_data)) S4Vectors::DataFrame(row.names = colnames(x)) else col_data,
    rowData = if (is.null(row_data)) S4Vectors::DataFrame(row.names = rownames(x)) else row_data
  )
  new("OmicsExperiment", se,
      omics_type = as.character(omics_type),
      processing_log = list(list(step = "init", time = Sys.time(), hash = NA_character_)),
      feature_map = S4Vectors::DataFrame())
}
