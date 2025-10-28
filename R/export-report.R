#' Export project report via Quarto (if available)
#' @param op OmniProject
#' @param out output file
#' @param template path to qmd template
#' @param params named list of parameters
#' @export
export_op_report <- function(op, out = "omni_report.html", template = system.file("scripts/default.qmd", package = "OmniOmicsR"),
                             params = list()) {
  params$summary <- list(
    assays = names(MultiAssayExperiment::experiments(op)),
    samples = nrow(MultiAssayExperiment::colData(op))
  )
  if (requireNamespace("quarto", quietly = TRUE)) {
    quarto::quarto_render(template, execute_params = params, output_file = out)
  } else if (requireNamespace("rmarkdown", quietly = TRUE)) {
    rmarkdown::render(input = template, output_file = out, params = params, envir = new.env(parent = globalenv()))
  } else {
    stop("Neither 'quarto' nor 'rmarkdown' is installed for report rendering.")
  }
  op@report <- normalizePath(out, mustWork = FALSE)
  invisible(out)
}

#' Save and load OmniProject/OmicsExperiment
#' @export
save_project <- function(x, file = "omni_project.rds") {
  saveRDS(x, file)
  invisible(file)
}

#' @export
load_project <- function(file) {
  readRDS(file)
}

#' Replay processing log (placeholder)
#' @param object OmicsExperiment
#' @export
replay <- function(object, from = 1, to = Inf, verbose = TRUE) {
  if (!is(object, "OmicsExperiment")) stop("replay currently supports OmicsExperiment logs")
  log <- object@processing_log
  if (length(log) == 0) return(object)
  steps <- log[seq.int(from, min(length(log), to))]
  # Start from the first raw assay when possible
  obj <- object
  for (st in steps) {
    fun <- as.character(st$step)
    args <- st
    args$step <- NULL
    args$time <- NULL
    if (verbose) message("Replaying: ", fun)
    if (exists(fun, mode = "function", inherits = TRUE)) {
      f <- get(fun, mode = "function")
      # Ensure object is the first argument
      res <- try(do.call(f, c(list(obj), args)), silent = TRUE)
      if (!inherits(res, "try-error")) obj <- res
    }
  }
  obj
}
