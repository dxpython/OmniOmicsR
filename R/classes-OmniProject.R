#' OmniProject class
#'
#' Wraps MultiAssayExperiment with project-level slots.
#' @slot design optional design matrix or list
#' @slot refdb character or list describing reference resources
#' @slot report character path to last rendered report
#' @exportClass OmniProject
setClass("OmniProject",
  contains = "MultiAssayExperiment",
  slots = c(
    design = "ANY",
    refdb = "ANY",
    report = "character"
  )
)

#' Show method for OmniProject
setMethod("show", "OmniProject", function(object) {
  assays <- names(MultiAssayExperiment::experiments(object))
  cat("OmniProject with assays: ", paste(assays, collapse = ", "), "\n", sep = "")
})

#' Build an OmniProject from a named list of OmicsExperiment
#' @param assays list of OmicsExperiment objects, named
#' @param design optional design info
#' @param refdb optional reference db specification
#' @return OmniProject
#' @export
as_op <- function(assays, design = NULL, refdb = NULL) {
  stopifnot(is.list(assays), length(assays) > 0)
  mae <- MultiAssayExperiment::MultiAssayExperiment(experiments = assays)
  new("OmniProject", mae, design = design, refdb = refdb, report = character(1))
}

