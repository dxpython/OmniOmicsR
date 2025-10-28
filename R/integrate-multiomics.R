#' Integrate multi-omics via DIABLO (mixOmics)
#'
#' @param op OmniProject
#' @param design optional design matrix
#' @param ncomp number of components
#' @param keepX optional list of selected variables per block
#' @param ... forwarded to mixOmics::block.splsda
#' @return fitted DIABLO object
#' @export
integrate_diablo <- function(op, design = NULL, ncomp = 2, keepX = NULL, ...) {
  if (!requireNamespace("mixOmics", quietly = TRUE)) {
    stop("mixOmics is not installed. Please install to use integrate_diablo().")
  }
  assays <- MultiAssayExperiment::experiments(op)
  X <- lapply(assays, function(oe) t(SummarizedExperiment::assay(oe, 1)))
  first <- assays[[1]]
  y <- SummarizedExperiment::colData(first)$group
  if (is.null(y)) stop("colData(first assay)$group is required for supervised DIABLO.")
  if (is.null(design)) design <- diag(length(X))
  fit <- mixOmics::block.splsda(X, Y = y, design = design, ncomp = ncomp, keepX = keepX, ...)
  attr(fit, "omni") <- TRUE
  fit
}

#' Integrate multi-omics via MOFA2
#'
#' @param op OmniProject
#' @param n_factors number of latent factors
#' @param scale_views logical
#' @param ... extra options for prepare/run
#' @return Trained MOFA model
#' @export
integrate_mofa <- function(op, n_factors = 5, scale_views = TRUE, ...) {
  if (!requireNamespace("MOFA2", quietly = TRUE)) stop("MOFA2 not installed")
  assays <- MultiAssayExperiment::experiments(op)
  X <- lapply(assays, function(oe) t(SummarizedExperiment::assay(oe, 1)))
  data <- list()
  for (nm in names(X)) data[[nm]] <- X[[nm]]
  model <- MOFA2::create_mofa(data)
  opts <- MOFA2::get_default_training_options(model)
  data_opts <- MOFA2::get_default_data_options(model)
  data_opts$scale_views <- scale_views
  model <- MOFA2::prepare_mofa(model, data_options = data_opts, training_options = opts, model_options = list(factors = n_factors))
  model <- MOFA2::run_mofa(model, ...)  # may take time on large data
  attr(model, "omni") <- TRUE
  model
}

#' Integrate via RGCCA (regularized GCCA)
#' @param op OmniProject
#' @param scheme RGCCA scheme (e.g., 'centroid')
#' @param tau regularization per block (NULL for automatic)
#' @param scale logical
#' @param ncomp number of components
#' @export
integrate_rgcca <- function(op, scheme = "centroid", tau = NULL, scale = TRUE, ncomp = 2) {
  if (!requireNamespace("RGCCA", quietly = TRUE)) stop("RGCCA not installed")
  assays <- MultiAssayExperiment::experiments(op)
  blocks <- lapply(assays, function(oe) t(SummarizedExperiment::assay(oe, 1)))
  rg <- RGCCA::rgcca(blocks, scheme = scheme, tau = tau, scale = scale, ncomp = rep(ncomp, length(blocks)))
  attr(rg, "omni") <- TRUE
  rg
}

#' Classical canonical correlation between two assays
#' @param x,y OmicsExperiment
#' @param assay_x,assay_y assay selection
#' @param ncomp components to compute
#' @export
integrate_canonical <- function(x, y, assay_x = 1, assay_y = 1, ncomp = 2) {
  X <- t(SummarizedExperiment::assay(x, assay_x))
  Y <- t(SummarizedExperiment::assay(y, assay_y))
  common <- intersect(rownames(X), rownames(Y))
  if (length(common) < 3) stop("Need >=3 shared samples for CCA")
  X <- scale(X[common, , drop = FALSE])
  Y <- scale(Y[common, , drop = FALSE])
  cc <- stats::cancor(X, Y)
  list(cancor = cc$cor[seq_len(min(ncomp, length(cc$cor)))], xcoef = cc$xcoef, ycoef = cc$ycoef)
}
