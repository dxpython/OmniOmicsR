#' Bayesian inference for omics data
#'
#' @description
#' Bayesian hierarchical models, differential expression, and parameter estimation

#' Bayesian differential expression using Stan
#' @param oe OmicsExperiment
#' @param design formula or design matrix
#' @param contrast character, contrast of interest
#' @param prior_sd numeric, prior standard deviation for coefficients
#' @param chains integer, number of MCMC chains
#' @param iter integer, iterations per chain
#' @param ... additional Stan parameters
#' @return list with posterior samples and summaries
#' @export
bayesian_dea <- function(oe, design, contrast, prior_sd = 2,
                         chains = 4, iter = 2000, ...) {

  if (!requireNamespace("rstan", quietly = TRUE)) {
    warning("rstan not available, using empirical Bayes approximation")
    return(.empirical_bayes_dea(oe, design, contrast))
  }

  # Extract data
  counts <- SummarizedExperiment::assay(oe, 1)
  col_data <- SummarizedExperiment::colData(oe)

  # Parse design
  if (inherits(design, "formula")) {
    design_mat <- model.matrix(design, data = as.data.frame(col_data))
  } else {
    design_mat <- as.matrix(design)
  }

  # Stan model for negative binomial
  stan_code <- "
  data {
    int<lower=0> N; // number of samples
    int<lower=0> P; // number of predictors
    int<lower=0> G; // number of genes
    int<lower=0> y[N, G]; // count matrix
    matrix[N, P] X; // design matrix
    real<lower=0> prior_sd; // prior SD
  }
  parameters {
    matrix[G, P] beta; // coefficients
    vector<lower=0>[G] phi; // dispersion
  }
  model {
    // Priors
    for (g in 1:G) {
      beta[g] ~ normal(0, prior_sd);
      phi[g] ~ gamma(2, 1);
    }

    // Likelihood
    for (g in 1:G) {
      vector[N] mu = exp(X * beta[g]');
      for (n in 1:N) {
        y[n, g] ~ neg_binomial_2(mu[n], phi[g]);
      }
    }
  }
  "

  # Prepare data for subset of genes (computational tractability)
  top_genes <- order(-rowMeans(counts))[1:min(100, nrow(counts))]
  counts_subset <- counts[top_genes, ]

  stan_data <- list(
    N = ncol(counts_subset),
    P = ncol(design_mat),
    G = nrow(counts_subset),
    y = t(counts_subset),
    X = design_mat,
    prior_sd = prior_sd
  )

  # Fit
  fit <- rstan::stan(
    model_code = stan_code,
    data = stan_data,
    chains = chains,
    iter = iter,
    ...
  )

  # Extract posteriors
  beta_samples <- rstan::extract(fit, "beta")$beta

  # Compute posterior probabilities for contrast
  # (simplified: assuming contrast is a coefficient index)
  contrast_idx <- as.integer(contrast)
  posterior_effects <- beta_samples[, , contrast_idx]

  # Summarize
  summaries <- data.frame(
    gene = rownames(counts_subset),
    posterior_mean = colMeans(posterior_effects),
    posterior_sd = apply(posterior_effects, 2, sd),
    prob_positive = colMeans(posterior_effects > 0),
    prob_negative = colMeans(posterior_effects < 0)
  )

  summaries$prob_de <- pmax(summaries$prob_positive, summaries$prob_negative)

  list(
    fit = fit,
    summaries = summaries,
    posterior_effects = posterior_effects,
    method = "stan_negbin"
  )
}

#' Empirical Bayes fallback (using limma)
#' @keywords internal
.empirical_bayes_dea <- function(oe, design, contrast) {

  if (!requireNamespace("limma", quietly = TRUE)) {
    stop("Neither rstan nor limma available")
  }

  counts <- SummarizedExperiment::assay(oe, 1)
  col_data <- SummarizedExperiment::colData(oe)

  if (inherits(design, "formula")) {
    design_mat <- model.matrix(design, data = as.data.frame(col_data))
  } else {
    design_mat <- as.matrix(design)
  }

  # Log-transform and voom
  y <- edgeR::DGEList(counts = counts)
  y <- edgeR::calcNormFactors(y)
  v <- limma::voom(y, design_mat)

  # Fit empirical Bayes
  fit <- limma::lmFit(v, design_mat)
  fit <- limma::eBayes(fit)

  # Results for contrast
  results <- limma::topTable(fit, coef = contrast, number = Inf)

  list(
    fit = fit,
    summaries = data.frame(
      gene = rownames(results),
      posterior_mean = results$logFC,
      posterior_sd = sqrt(results$s2.post),
      prob_de = 1 - results$P.Value
    ),
    method = "empirical_bayes_limma"
  )
}

#' Bayesian network inference
#' @param oe OmicsExperiment
#' @param method character, method for network learning
#' @param score character, scoring function
#' @param max_parents integer, max parents per node
#' @return list with learned network
#' @export
bayesian_network <- function(oe, method = "hc", score = "bic", max_parents = 3) {

  if (!requireNamespace("bnlearn", quietly = TRUE)) {
    warning("bnlearn not available, using correlation network")
    return(.correlation_network(oe))
  }

  # Extract data (use top variable features)
  x <- t(SummarizedExperiment::assay(oe, 1))
  x[is.na(x)] <- 0

  # Select top variable features (for computational tractability)
  vars <- apply(x, 2, var)
  top_idx <- order(-vars)[1:min(50, ncol(x))]
  x <- x[, top_idx]

  # Discretize for Bayesian network learning
  x_discrete <- apply(x, 2, function(col) {
    cut(col, breaks = 3, labels = c("low", "medium", "high"))
  })

  df <- as.data.frame(x_discrete)

  # Learn structure
  if (method == "hc") {
    bn <- bnlearn::hc(df, score = score, maxp = max_parents)
  } else if (method == "tabu") {
    bn <- bnlearn::tabu(df, score = score, maxp = max_parents)
  } else {
    bn <- bnlearn::gs(df)
  }

  # Fit parameters
  fitted <- bnlearn::bn.fit(bn, df)

  # Extract edges
  edges <- bnlearn::arcs(bn)

  list(
    network = bn,
    fitted = fitted,
    edges = edges,
    n_edges = nrow(edges),
    method = method,
    score = score
  )
}

#' Correlation network fallback
#' @keywords internal
.correlation_network <- function(oe, threshold = 0.3) {

  x <- t(SummarizedExperiment::assay(oe, 1))
  x[is.na(x)] <- 0

  # Top variable features
  vars <- apply(x, 2, var)
  top_idx <- order(-vars)[1:min(50, ncol(x))]
  x <- x[, top_idx]

  # Correlation matrix
  cor_mat <- cor(x)
  cor_mat[abs(cor_mat) < threshold] <- 0
  diag(cor_mat) <- 0

  # Extract edges
  edges_idx <- which(cor_mat != 0, arr.ind = TRUE)
  edges_idx <- edges_idx[edges_idx[, 1] < edges_idx[, 2], ]  # Upper triangle

  edges <- data.frame(
    from = colnames(x)[edges_idx[, 1]],
    to = colnames(x)[edges_idx[, 2]],
    correlation = cor_mat[edges_idx]
  )

  list(
    network = NULL,
    fitted = NULL,
    edges = edges,
    n_edges = nrow(edges),
    method = "correlation_fallback",
    correlation_matrix = cor_mat
  )
}

#' Bayesian meta-analysis
#' @param effect_list list of effect sizes from multiple studies
#' @param se_list list of standard errors
#' @param method character, "random" or "fixed" effects
#' @return list with meta-analysis results
#' @export
bayesian_meta_analysis <- function(effect_list, se_list, method = "random") {

  if (!requireNamespace("metafor", quietly = TRUE)) {
    stop("metafor package required")
  }

  # Combine studies
  k <- length(effect_list)
  n_features <- length(effect_list[[1]])

  results <- lapply(1:n_features, function(i) {
    yi <- sapply(effect_list, function(x) x[i])
    sei <- sapply(se_list, function(x) x[i])

    # Remove NA
    valid <- !is.na(yi) & !is.na(sei)
    yi <- yi[valid]
    sei <- sei[valid]

    if (length(yi) < 2) {
      return(data.frame(
        estimate = NA, se = NA, ci_lower = NA, ci_upper = NA,
        pval = NA, tau2 = NA, I2 = NA
      ))
    }

    # Meta-analysis
    res <- metafor::rma(yi, sei, method = if (method == "random") "REML" else "FE")

    data.frame(
      estimate = res$beta[1],
      se = res$se,
      ci_lower = res$ci.lb,
      ci_upper = res$ci.ub,
      pval = res$pval,
      tau2 = if (method == "random") res$tau2 else 0,
      I2 = if (method == "random") res$I2 else 0
    )
  })

  results_df <- do.call(rbind, results)
  results_df$feature <- names(effect_list[[1]])

  list(
    results = results_df,
    n_studies = k,
    n_features = n_features,
    method = method
  )
}
