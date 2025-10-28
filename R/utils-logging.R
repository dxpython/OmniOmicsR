#' Append a processing log entry
#' @keywords internal
.add_log <- function(object, step, params = list()) {
  entry <- c(list(step = step, time = Sys.time()), params)
  if (is(object, "OmicsExperiment")) {
    object@processing_log <- c(object@processing_log, list(entry))
  }
  object
}

