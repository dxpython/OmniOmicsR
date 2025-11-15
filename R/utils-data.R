#' Data Extraction and Preparation Utilities
#'
#' Internal utility functions for common data extraction and preparation tasks
#' @name utils-data
#' @keywords internal
NULL

#' Extract data matrix from OmicsExperiment or matrix
#'
#' @param oe OmicsExperiment, SummarizedExperiment, or matrix
#' @param assay_name character, name of assay to extract (default: first assay)
#' @param transpose logical, transpose the matrix (default: TRUE for samples x features)
#' @param remove_na logical, replace NA values (default: TRUE)
#' @param na_value numeric, value to replace NAs with (default: 0)
#' @param scale logical, scale columns to mean=0, sd=1 (default: FALSE)
#' @return numeric matrix
#' @keywords internal
.extract_data_matrix <- function(oe, assay_name = NULL, transpose = TRUE,
                                 remove_na = TRUE, na_value = 0, scale = FALSE) {

  # Extract assay
  if (inherits(oe, "SummarizedExperiment")) {
    if (is.null(assay_name)) {
      x <- SummarizedExperiment::assay(oe, 1)
    } else {
      x <- SummarizedExperiment::assay(oe, assay_name)
    }
  } else if (is.matrix(oe)) {
    x <- oe
  } else if (is.data.frame(oe)) {
    x <- as.matrix(oe)
  } else {
    stop("oe must be SummarizedExperiment, matrix, or data.frame")
  }

  # Ensure numeric
  storage.mode(x) <- "double"

  # Transpose if requested (features x samples -> samples x features)
  if (transpose) x <- t(x)

  # Handle NAs
  if (remove_na) x[is.na(x)] <- na_value

  # Scale if requested
  if (scale) {
    x <- scale(x)
    if (remove_na) x[is.na(x)] <- 0  # scale() can introduce NAs
  }

  x
}

#' Extract feature names from OmicsExperiment or matrix
#'
#' @param oe OmicsExperiment, SummarizedExperiment, or matrix
#' @param assay_name character, assay name (if OmicsExperiment)
#' @return character vector of feature names
#' @keywords internal
.extract_feature_names <- function(oe, assay_name = NULL) {
  if (inherits(oe, "SummarizedExperiment")) {
    if (is.null(assay_name)) {
      rownames(SummarizedExperiment::assay(oe, 1))
    } else {
      rownames(SummarizedExperiment::assay(oe, assay_name))
    }
  } else {
    rownames(as.matrix(oe))
  }
}

#' Extract sample names from OmicsExperiment or matrix
#'
#' @param oe OmicsExperiment, SummarizedExperiment, or matrix
#' @param assay_name character, assay name (if OmicsExperiment)
#' @return character vector of sample names
#' @keywords internal
.extract_sample_names <- function(oe, assay_name = NULL) {
  if (inherits(oe, "SummarizedExperiment")) {
    if (is.null(assay_name)) {
      colnames(SummarizedExperiment::assay(oe, 1))
    } else {
      colnames(SummarizedExperiment::assay(oe, assay_name))
    }
  } else {
    colnames(as.matrix(oe))
  }
}

#' Require package or use fallback function
#'
#' @param package character, package name
#' @param fallback_fn function to call if package not available
#' @param ... arguments passed to fallback_fn
#' @param warning_msg character, custom warning message
#' @param quietly logical, suppress namespace loading messages
#' @return result from main operation or fallback_fn
#' @keywords internal
.require_or_fallback <- function(package, fallback_fn, ...,
                                 warning_msg = NULL, quietly = TRUE) {

  if (!requireNamespace(package, quietly = quietly)) {
    if (is.null(warning_msg)) {
      warning_msg <- paste0(package, " not available, using fallback method")
    }
    warning(warning_msg)
    return(fallback_fn(...))
  }

  NULL  # Package is available, proceed with main function
}

#' Validate matrix dimensions match
#'
#' @param x matrix or data.frame
#' @param y matrix, vector, or data.frame
#' @param dim_x integer, which dimension of x to check (1=rows, 2=cols)
#' @param dim_y integer, which dimension of y to check
#' @param x_name character, name of x for error message
#' @param y_name character, name of y for error message
#' @return invisible TRUE if valid
#' @keywords internal
.validate_dimensions <- function(x, y, dim_x = 2, dim_y = 1,
                                 x_name = "x", y_name = "y") {

  # Get dimensions
  if (is.vector(y)) {
    dim_y_val <- length(y)
  } else {
    dim_y_val <- dim(y)[dim_y]
  }

  dim_x_val <- dim(x)[dim_x]

  # Check match
  if (dim_x_val != dim_y_val) {
    stop(sprintf(
      "Dimension mismatch: %s (dim %d = %d) does not match %s (dim %d = %d)",
      x_name, dim_x, dim_x_val,
      y_name, dim_y, dim_y_val
    ))
  }

  invisible(TRUE)
}

#' Validate input is non-negative matrix
#'
#' @param x matrix to validate
#' @param name character, variable name for error message
#' @param allow_na logical, allow NA values
#' @return invisible TRUE if valid
#' @keywords internal
.validate_nonnegative_matrix <- function(x, name = "x", allow_na = TRUE) {

  if (!is.matrix(x) && !is.data.frame(x)) {
    stop(sprintf("%s must be a matrix or data.frame", name))
  }

  if (!is.numeric(as.matrix(x)[1, 1])) {
    stop(sprintf("%s must contain numeric values", name))
  }

  if (!allow_na && any(is.na(x))) {
    stop(sprintf("%s contains NA values (not allowed)", name))
  }

  if (any(x < 0, na.rm = TRUE)) {
    stop(sprintf("%s contains negative values", name))
  }

  invisible(TRUE)
}

#' Validate feature names are unique
#'
#' @param x character vector of names
#' @param name character, variable name for error message
#' @param allow_null logical, allow NULL names
#' @return invisible TRUE if valid
#' @keywords internal
.validate_unique_names <- function(x, name = "names", allow_null = FALSE) {

  if (is.null(x)) {
    if (!allow_null) {
      stop(sprintf("%s cannot be NULL", name))
    }
    return(invisible(TRUE))
  }

  if (length(x) != length(unique(x))) {
    dups <- x[duplicated(x)]
    stop(sprintf(
      "%s contains %d duplicated values: %s",
      name,
      length(dups),
      paste(head(dups, 5), collapse = ", ")
    ))
  }

  invisible(TRUE)
}

#' Get number of samples from various input types
#'
#' @param oe OmicsExperiment, matrix, or data.frame
#' @return integer, number of samples
#' @keywords internal
.get_n_samples <- function(oe) {
  if (inherits(oe, "SummarizedExperiment")) {
    ncol(SummarizedExperiment::assay(oe, 1))
  } else {
    ncol(as.matrix(oe))
  }
}

#' Get number of features from various input types
#'
#' @param oe OmicsExperiment, matrix, or data.frame
#' @return integer, number of features
#' @keywords internal
.get_n_features <- function(oe) {
  if (inherits(oe, "SummarizedExperiment")) {
    nrow(SummarizedExperiment::assay(oe, 1))
  } else {
    nrow(as.matrix(oe))
  }
}

#' Select top variable features
#'
#' @param x numeric matrix (features x samples)
#' @param n_top integer, number of top features to select
#' @param method character, variance metric ("var", "mad", "cv")
#' @return integer vector of selected feature indices
#' @keywords internal
.select_top_variable_features <- function(x, n_top = 1000, method = "var") {

  if (method == "var") {
    scores <- apply(x, 1, var, na.rm = TRUE)
  } else if (method == "mad") {
    scores <- apply(x, 1, mad, na.rm = TRUE)
  } else if (method == "cv") {
    means <- rowMeans(x, na.rm = TRUE)
    sds <- apply(x, 1, sd, na.rm = TRUE)
    scores <- sds / (means + 1e-8)  # Coefficient of variation
  } else {
    stop("method must be 'var', 'mad', or 'cv'")
  }

  scores[is.na(scores)] <- 0
  order(-scores)[1:min(n_top, length(scores))]
}

#' Safe matrix subset that preserves dimensions
#'
#' @param x matrix
#' @param rows integer vector or logical vector
#' @param cols integer vector or logical vector
#' @return matrix (always 2D, even if single row/col)
#' @keywords internal
.safe_subset_matrix <- function(x, rows = NULL, cols = NULL) {

  if (is.null(rows)) rows <- seq_len(nrow(x))
  if (is.null(cols)) cols <- seq_len(ncol(x))

  # Ensure we get a matrix, not a vector
  result <- x[rows, cols, drop = FALSE]

  # Preserve names
  if (!is.null(rownames(x))) rownames(result) <- rownames(x)[rows]
  if (!is.null(colnames(x))) colnames(result) <- colnames(x)[cols]

  result
}
