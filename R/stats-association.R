#' Feature-wise correlation across assays or with covariates
#'
#' Compute correlations between two matrices (features x samples) matched by samples, or
#' between features and a numeric covariate in colData.
#' @param x OmicsExperiment or numeric matrix (features x samples)
#' @param y OmicsExperiment, numeric matrix, or character name of a numeric covariate in colData(x)
#' @param method 'pearson' or 'spearman'
#' @param assay_x,assay_y assay names/indices when x/y are OmicsExperiment
#' @param top return top N strongest absolute correlations (optional)
#' @return data.frame with feature_x, feature_y (or covariate), cor, pval (approx via t-test)
#' @export
assoc_corr <- function(x, y, method = c("pearson","spearman"), assay_x = 1, assay_y = 1, top = NULL) {
  method <- match.arg(method)
  # Resolve matrices
  if (methods::is(x, "OmicsExperiment")) xmat <- SummarizedExperiment::assay(x, assay_x) else xmat <- as.matrix(x)
  if (is.character(y) && length(y) == 1 && methods::is(x, "OmicsExperiment")) {
    cd <- as.data.frame(SummarizedExperiment::colData(x))
    if (!y %in% names(cd)) stop("Covariate not found in colData: ", y)
    yvec <- as.numeric(cd[[y]])
    if (anyNA(yvec)) stop("Covariate contains NA; please clean it")
    # correlate each feature vs covariate
    corv <- apply(xmat, 1, function(v){ suppressWarnings(suppressMessages(stats::cor(v, yvec, method = method, use = "pairwise.complete.obs"))) })
    n <- sum(is.finite(yvec))
    tval <- corv * sqrt((n - 2)/(pmax(1 - corv^2, .Machine$double.eps)))
    pval <- 2 * stats::pt(-abs(tval), df = n - 2)
    df <- data.frame(feature_x = rownames(xmat), feature_y = y, cor = corv, pval = pval, stringsAsFactors = FALSE)
  } else {
    if (methods::is(y, "OmicsExperiment")) ymat <- SummarizedExperiment::assay(y, assay_y) else ymat <- as.matrix(y)
    common <- intersect(colnames(xmat), colnames(ymat))
    if (length(common) < 3) stop("Need >=3 shared samples to compute correlation")
    xmat <- xmat[, common, drop = FALSE]
    ymat <- ymat[, common, drop = FALSE]
    # compute all-vs-all correlations efficiently
    xc <- t(scale(t(xmat), center = TRUE, scale = TRUE))
    yc <- t(scale(t(ymat), center = TRUE, scale = TRUE))
    cor_mat <- (xc %*% t(yc)) / (length(common) - 1)
    # approximate p-values using t distribution per pair
    r <- as.vector(cor_mat)
    n <- length(common)
    tval <- r * sqrt((n - 2)/(pmax(1 - r^2, .Machine$double.eps)))
    pval <- 2 * stats::pt(-abs(tval), df = n - 2)
    df <- data.frame(
      feature_x = rep(rownames(xmat), times = nrow(ymat)),
      feature_y = rep(rownames(ymat), each = nrow(xmat)),
      cor = r,
      pval = pval,
      stringsAsFactors = FALSE
    )
  }
  if (!is.null(top)) {
    ord <- order(-abs(df$cor))
    df <- df[ord[seq_len(min(top, nrow(df)))], , drop = FALSE]
  }
  df
}

#' Linear mixed model association per feature
#'
#' Fits y ~ x + (1|random) for each feature using lme4::lmer when available.
#' @param object OmicsExperiment
#' @param covar column name in colData for fixed effect (numeric or factor)
#' @param random optional column name for random intercept
#' @param assay assay index/name
#' @return data.frame with effect, t-value, p-value (Satterthwaite approximations not provided)
#' @export
assoc_lmm <- function(object, covar, random = NULL, assay = 1) {
  if (!requireNamespace("lme4", quietly = TRUE)) stop("lme4 not installed")
  x <- SummarizedExperiment::assay(object, assay)
  cd <- as.data.frame(SummarizedExperiment::colData(object))
  if (!covar %in% names(cd)) stop("covariate not found: ", covar)
  if (!is.null(random) && !random %in% names(cd)) stop("random effect not found: ", random)
  res <- lapply(seq_len(nrow(x)), function(i){
    df <- data.frame(y = as.numeric(x[i, ]), cd, check.names = FALSE)
    if (is.null(random)) {
      fit <- lme4::lmer(stats::as.formula(paste0("y ~ ", covar, " + (1|.dummy)")), data = transform(df, .dummy = 1), REML = FALSE)
    } else {
      fit <- lme4::lmer(stats::as.formula(paste0("y ~ ", covar, " + (1|", random, ")")), data = df, REML = FALSE)
    }
    co <- summary(fit)$coefficients
    if (covar %in% rownames(co)) {
      c(effect = unname(co[covar, 1]), t = unname(co[covar, 3]), p = 2 * stats::pnorm(-abs(co[covar, 3])))
    } else {
      c(effect = NA_real_, t = NA_real_, p = NA_real_)
    }
  })
  out <- do.call(rbind, res)
  out <- as.data.frame(out)
  out$feature <- rownames(x)
  out
}

