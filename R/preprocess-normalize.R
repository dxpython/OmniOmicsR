#' TMM-like normalization (with edgeR if available)
#'
#' If edgeR is available, uses calcNormFactors and returns logCPM.
#' Otherwise falls back to library size scaling and log1p.
#' Output is stored as an assay named 'norm'.
#' @param object OmicsExperiment
#' @param assay which assay (name or index)
#' @return OmicsExperiment
#' @export
normalize_tmm <- function(object, assay = 1) {
  stopifnot(is(object, "OmicsExperiment"))
  mat <- SummarizedExperiment::assay(object, assay)
  if (requireNamespace("edgeR", quietly = TRUE)) {
    d <- edgeR::DGEList(counts = mat)
    d <- edgeR::calcNormFactors(d)
    norm <- edgeR::cpm(d, log = TRUE, prior.count = 1)
  } else {
    lib <- colSums(mat, na.rm = TRUE)
    lib[lib == 0] <- 1
    scaled <- sweep(mat, 2, lib / median(lib), "/")
    norm <- log1p(scaled)
  }
  SummarizedExperiment::assays(object)[["norm"]] <- as.matrix(norm)
  object <- .add_log(object, "normalize_tmm", list(assay = assay))
  object
}

#' Quantile normalization (limma/preprocessCore)
#' @param object OmicsExperiment
#' @param assay assay index/name
#' @param log_transform whether to log2(x+1) before normalization when data are counts/intensities
#' @export
normalize_quantile <- function(object, assay = 1, log_transform = TRUE) {
  x <- SummarizedExperiment::assay(object, assay)
  if (log_transform) xw <- log2(x + 1) else xw <- x
  if (requireNamespace("limma", quietly = TRUE)) {
    xn <- limma::normalizeBetweenArrays(xw, method = "quantile")
  } else if (requireNamespace("preprocessCore", quietly = TRUE)) {
    xn <- preprocessCore::normalize.quantiles(as.matrix(xw))
    rownames(xn) <- rownames(xw); colnames(xn) <- colnames(xw)
  } else {
    warning("Neither limma nor preprocessCore available; returning input.")
    xn <- xw
  }
  SummarizedExperiment::assays(object)[["qnorm"]] <- as.matrix(xn)
  object <- .add_log(object, "normalize_quantile", list(assay = assay, log = log_transform))
  object
}

#' Variance Stabilizing Transform (DESeq2)
#' @param object OmicsExperiment with counts assay
#' @param assay counts assay
#' @export
normalize_vst <- function(object, assay = 1) {
  if (!requireNamespace("DESeq2", quietly = TRUE)) stop("DESeq2 not installed")
  x <- SummarizedExperiment::assay(object, assay)
  cd <- as.data.frame(SummarizedExperiment::colData(object))
  dds <- DESeq2::DESeqDataSetFromMatrix(countData = round(x), colData = cd, design = ~ 1)
  vs <- DESeq2::vst(dds, blind = TRUE)
  SummarizedExperiment::assays(object)[["vst"]] <- as.matrix(SummarizedExperiment::assay(vs))
  object <- .add_log(object, "normalize_vst", list(assay = assay))
  object
}

#' Regularized log transform (DESeq2)
#' @param object OmicsExperiment with counts assay
#' @param assay counts assay
#' @export
normalize_rlog <- function(object, assay = 1) {
  if (!requireNamespace("DESeq2", quietly = TRUE)) stop("DESeq2 not installed")
  x <- SummarizedExperiment::assay(object, assay)
  cd <- as.data.frame(SummarizedExperiment::colData(object))
  dds <- DESeq2::DESeqDataSetFromMatrix(countData = round(x), colData = cd, design = ~ 1)
  rl <- DESeq2::rlog(dds, blind = TRUE)
  SummarizedExperiment::assays(object)[["rlog"]] <- as.matrix(SummarizedExperiment::assay(rl))
  object <- .add_log(object, "normalize_rlog", list(assay = assay))
  object
}

#' Minimal left-censored imputation (minProb-like)
#' @param object OmicsExperiment
#' @param assay which assay to impute
#' @param shift numeric, mean shift below observed
#' @export
impute_minprob <- function(object, assay = 1, shift = 1.8) {
  stopifnot(is(object, "OmicsExperiment"))
  x <- SummarizedExperiment::assay(object, assay)
  imputed <- x
  for (i in seq_len(nrow(x))) {
    row <- x[i, ]
    nas <- is.na(row) | is.infinite(row)
    if (any(!nas)) {
      mu <- mean(row[!nas], na.rm = TRUE)
      sdv <- stats::sd(row[!nas], na.rm = TRUE)
      if (!is.finite(sdv) || sdv == 0) sdv <- 0.1
      draw <- rnorm(sum(nas), mean = mu - shift * sdv, sd = sdv * 0.3)
      imputed[i, nas] <- draw
    }
  }
  SummarizedExperiment::assays(object)[["imputed"]] <- imputed
  object <- .add_log(object, "impute_minprob", list(assay = assay, shift = shift))
  object
}

#' KNN imputation (impute::impute.knn)
#' @param object OmicsExperiment
#' @param assay assay to impute
#' @param k number of neighbors
#' @export
impute_knn <- function(object, assay = 1, k = 10) {
  if (!requireNamespace("impute", quietly = TRUE)) {
    warning("'impute' package not installed; falling back to column median imputation.")
    x <- SummarizedExperiment::assay(object, assay)
    for (j in seq_len(ncol(x))) {
      nas <- is.na(x[, j]) | is.infinite(x[, j])
      if (any(nas)) {
        med <- stats::median(x[, j], na.rm = TRUE)
        x[nas, j] <- med
      }
    }
    SummarizedExperiment::assays(object)[["imputed_knn"]] <- x
    return(.add_log(object, "impute_knn", list(assay = assay, k = k, fallback = TRUE)))
  }
  x <- SummarizedExperiment::assay(object, assay)
  imp <- impute::impute.knn(as.matrix(x), k = k)
  SummarizedExperiment::assays(object)[["imputed_knn"]] <- imp$data
  .add_log(object, "impute_knn", list(assay = assay, k = k))
}
