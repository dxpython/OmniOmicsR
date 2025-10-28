#' Batch correction via ComBat (if available)
#'
#' Uses sva::ComBat if installed. Writes a new assay named 'combat'.
#' @param object OmicsExperiment
#' @param batch column name in colData indicating batch
#' @param assay assay index or name to correct
#' @param covariates optional character vector of covariate column names
#' @param par.prior logical for ComBat parametric prior
#' @param BPPARAM parallel backend
#' @export
batch_combat <- function(object, batch, assay = 1, covariates = NULL, par.prior = TRUE,
                         BPPARAM = .omni_bpparam()) {
  stopifnot(is(object, "OmicsExperiment"))
  if (!requireNamespace("sva", quietly = TRUE)) {
    warning("sva not installed; batch_combat is a no-op.")
    return(object)
  }
  x <- SummarizedExperiment::assay(object, assay)
  cd <- as.data.frame(SummarizedExperiment::colData(object))
  if (!batch %in% colnames(cd)) stop("batch column not found in colData: ", batch)
  mod <- if (!is.null(covariates) && length(covariates) > 0) stats::model.matrix(~ ., cd[, covariates, drop = FALSE]) else NULL
  adj <- sva::ComBat(dat = x, batch = cd[[batch]], mod = mod, par.prior = par.prior)
  SummarizedExperiment::assays(object)[["combat"]] <- as.matrix(adj)
  object <- .add_log(object, "batch_combat", list(batch = batch, assay = assay))
  object
}

#' Batch correction via fastMNN (batchelor)
#'
#' Produces a corrected expression matrix using mnnCorrect (expression-level) when available,
#' otherwise returns unchanged object with a warning.
#' @param object OmicsExperiment
#' @param batch batch column in colData
#' @param assay assay index/name
#' @export
batch_mnn <- function(object, batch, assay = 1) {
  if (!requireNamespace("batchelor", quietly = TRUE)) {
    warning("batchelor not installed; batch_mnn is a no-op.")
    return(object)
  }
  x <- SummarizedExperiment::assay(object, assay)
  cd <- as.data.frame(SummarizedExperiment::colData(object))
  if (!batch %in% colnames(cd)) stop("batch column not found in colData: ", batch)
  batches <- split(seq_len(ncol(x)), cd[[batch]])
  mats <- lapply(batches, function(ix) x[, ix, drop = FALSE])
  corr <- batchelor::mnnCorrect(mats, subset.row = rownames(x))
  # mnnCorrect may return SummarizedExperiment; extract corrected assay
  if (methods::is(corr, "SummarizedExperiment")) {
    adj <- SummarizedExperiment::assay(corr, "corrected")
  } else if (is.list(corr) && !is.null(corr$corrected)) {
    adj <- corr$corrected
  } else {
    stop("Unexpected output from mnnCorrect")
  }
  # Columns align to concatenation of batches; reorder to original sample order
  new_order <- unlist(batches, use.names = FALSE)
  adj <- adj[, order(match(new_order, seq_len(ncol(x)))), drop = FALSE]
  colnames(adj) <- colnames(x)
  SummarizedExperiment::assays(object)[["mnn"]] <- as.matrix(adj)
  .add_log(object, "batch_mnn", list(batch = batch, assay = assay))
}

#' Harmony correction on PCA embeddings
#'
#' Runs PCA on the specified assay, then applies harmony to remove batch effects
#' in the low-dimensional space. Stores harmonized embeddings in colData as HARMONY1..k.
#' @param object OmicsExperiment
#' @param batch batch column in colData
#' @param assay assay index/name
#' @param n_pcs number of PCs to use
#' @export
batch_harmony <- function(object, batch, assay = 1, n_pcs = 20) {
  if (!requireNamespace("harmony", quietly = TRUE)) {
    warning("harmony not installed; batch_harmony is a no-op.")
    return(object)
  }
  x <- SummarizedExperiment::assay(object, assay)
  x <- t(scale(t(x)))
  p <- stats::prcomp(t(x), rank. = n_pcs)
  emb <- p$x
  cd <- as.data.frame(SummarizedExperiment::colData(object))
  if (!batch %in% colnames(cd)) stop("batch column not found in colData: ", batch)
  harm <- harmony::HarmonyMatrix(data_mat = t(emb), meta_data = cd, vars_use = batch)
  harm <- t(harm)
  # store embeddings into colData
  for (i in seq_len(ncol(harm))) {
    SummarizedExperiment::colData(object)[[paste0("HARMONY", i)]] <- harm[, i]
  }
  .add_log(object, "batch_harmony", list(batch = batch, assay = assay, n_pcs = n_pcs))
}
