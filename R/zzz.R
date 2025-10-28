#' OmniOmicsR
#'
#' Package startup options and utilities.
#' @keywords internal
"_PACKAGE"

.onLoad <- function(libname, pkgname) {
  op <- options()
  op.omni <- list(
    omni.verbose = TRUE
  )
  toset <- !(names(op.omni) %in% names(op))
  if (any(toset)) options(op.omni[toset])
  invisible()
}

