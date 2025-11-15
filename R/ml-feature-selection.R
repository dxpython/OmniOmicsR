#' Advanced feature selection methods
#'
#' @description
#' LASSO, elastic net, Boruta, stability selection for high-dimensional omics

#' LASSO/Elastic Net feature selection
#' @param oe OmicsExperiment or matrix (features x samples)
#' @param outcome vector of outcomes
#' @param alpha numeric, elastic net mixing (1=LASSO, 0=Ridge)
#' @param cv_folds integer, cross-validation folds
#' @param family character, glmnet family
#' @return list with selected features and model
#' @export
feature_select_elastic_net <- function(oe, outcome, alpha = 1, cv_folds = 10,
                                       family = NULL) {

  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("glmnet package required")
  }

  # Extract data
  if (inherits(oe, "SummarizedExperiment")) {
    x <- t(SummarizedExperiment::assay(oe, 1))
    feature_names <- rownames(SummarizedExperiment::assay(oe, 1))
  } else {
    x <- t(as.matrix(oe))
    feature_names <- rownames(as.matrix(oe))
  }

  x[is.na(x)] <- 0

  # Auto-detect family
  if (is.null(family)) {
    family <- if (is.factor(outcome) || is.character(outcome)) "binomial" else "gaussian"
  }

  # CV to find optimal lambda
  cv_fit <- glmnet::cv.glmnet(
    x = x,
    y = outcome,
    alpha = alpha,
    nfolds = cv_folds,
    family = family,
    standardize = TRUE
  )

  # Get coefficients at optimal lambda
  coefs <- as.matrix(coef(cv_fit, s = "lambda.min"))
  selected_idx <- which(coefs[-1, 1] != 0)  # Exclude intercept

  selected_features <- data.frame(
    feature = feature_names[selected_idx],
    coefficient = coefs[selected_idx + 1, 1],  # +1 for intercept
    abs_coefficient = abs(coefs[selected_idx + 1, 1])
  )

  selected_features <- selected_features[order(-selected_features$abs_coefficient), ]

  list(
    model = cv_fit,
    selected_features = selected_features,
    n_selected = nrow(selected_features),
    lambda_min = cv_fit$lambda.min,
    lambda_1se = cv_fit$lambda.1se,
    alpha = alpha,
    family = family
  )
}

#' Boruta feature selection (all-relevant)
#' @param oe OmicsExperiment or matrix
#' @param outcome vector of outcomes
#' @param max_runs integer, maximum Boruta iterations
#' @param p_value numeric, significance threshold
#' @return list with selected features
#' @export
feature_select_boruta <- function(oe, outcome, max_runs = 100, p_value = 0.01) {

  if (!requireNamespace("Boruta", quietly = TRUE)) {
    warning("Boruta package not available, using correlation-based selection")
    return(.feature_select_correlation(oe, outcome, p_value = p_value))
  }

  # Extract data
  if (inherits(oe, "SummarizedExperiment")) {
    x <- t(SummarizedExperiment::assay(oe, 1))
    feature_names <- rownames(SummarizedExperiment::assay(oe, 1))
  } else {
    x <- t(as.matrix(oe))
    feature_names <- rownames(as.matrix(oe))
  }

  x[is.na(x)] <- 0

  # Run Boruta
  df <- as.data.frame(x)
  colnames(df) <- make.names(feature_names)
  df$outcome <- outcome

  boruta_result <- Boruta::Boruta(
    outcome ~ .,
    data = df,
    maxRuns = max_runs,
    pValue = p_value
  )

  # Get selected features
  decisions <- Boruta::attStats(boruta_result)
  confirmed <- rownames(decisions)[decisions$decision == "Confirmed"]
  tentative <- rownames(decisions)[decisions$decision == "Tentative"]

  confirmed_df <- data.frame(
    feature = confirmed,
    importance = decisions[confirmed, "meanImp"],
    decision = "Confirmed"
  )

  tentative_df <- data.frame(
    feature = tentative,
    importance = decisions[tentative, "meanImp"],
    decision = "Tentative"
  )

  selected_features <- rbind(confirmed_df, tentative_df)
  selected_features <- selected_features[order(-selected_features$importance), ]

  list(
    boruta_result = boruta_result,
    selected_features = selected_features,
    n_confirmed = nrow(confirmed_df),
    n_tentative = nrow(tentative_df),
    all_stats = decisions
  )
}

#' Correlation-based fallback feature selection
#' @keywords internal
.feature_select_correlation <- function(oe, outcome, p_value = 0.01, top_n = 100) {

  if (inherits(oe, "SummarizedExperiment")) {
    x <- t(SummarizedExperiment::assay(oe, 1))
    feature_names <- rownames(SummarizedExperiment::assay(oe, 1))
  } else {
    x <- t(as.matrix(oe))
    feature_names <- rownames(as.matrix(oe))
  }

  x[is.na(x)] <- 0

  # Convert outcome to numeric if needed
  if (is.factor(outcome)) {
    y_numeric <- as.numeric(outcome)
  } else {
    y_numeric <- outcome
  }

  # Compute correlations
  cors <- apply(x, 2, function(col) {
    if (sd(col) == 0) return(c(cor = 0, p = 1))
    test <- cor.test(col, y_numeric)
    c(cor = test$estimate, p = test$p.value)
  })

  cors <- t(cors)
  selected_idx <- which(cors[, "p.p-value"] < p_value)

  selected_features <- data.frame(
    feature = feature_names[selected_idx],
    correlation = cors[selected_idx, "cor.cor"],
    p_value = cors[selected_idx, "p.p-value"]
  )

  selected_features <- selected_features[order(selected_features$p_value), ]
  selected_features <- head(selected_features, top_n)

  list(
    selected_features = selected_features,
    n_selected = nrow(selected_features),
    method = "correlation_fallback"
  )
}

#' Stability selection
#' @param oe OmicsExperiment or matrix
#' @param outcome vector of outcomes
#' @param n_bootstrap integer, number of bootstrap samples
#' @param threshold numeric, stability threshold (0-1)
#' @param alpha numeric, elastic net parameter
#' @return list with stable features
#' @export
feature_select_stability <- function(oe, outcome, n_bootstrap = 100,
                                     threshold = 0.6, alpha = 1) {

  # Extract data
  if (inherits(oe, "SummarizedExperiment")) {
    x <- t(SummarizedExperiment::assay(oe, 1))
    feature_names <- rownames(SummarizedExperiment::assay(oe, 1))
  } else {
    x <- t(as.matrix(oe))
    feature_names <- rownames(as.matrix(oe))
  }

  x[is.na(x)] <- 0
  n_samples <- nrow(x)
  n_features <- ncol(x)

  # Track feature selection across bootstraps
  selection_matrix <- matrix(0, nrow = n_bootstrap, ncol = n_features)

  for (i in 1:n_bootstrap) {
    # Bootstrap sample
    boot_idx <- sample(1:n_samples, size = n_samples, replace = TRUE)
    boot_x <- x[boot_idx, ]
    boot_y <- outcome[boot_idx]

    # Run elastic net
    result <- feature_select_elastic_net(
      t(boot_x), boot_y, alpha = alpha, cv_folds = 5
    )

    # Mark selected features
    if (nrow(result$selected_features) > 0) {
      selected_names <- result$selected_features$feature
      selected_idx <- match(selected_names, feature_names)
      selection_matrix[i, selected_idx] <- 1
    }
  }

  # Compute stability scores
  stability_scores <- colMeans(selection_matrix)
  names(stability_scores) <- feature_names

  # Select stable features
  stable_idx <- which(stability_scores >= threshold)
  stable_features <- data.frame(
    feature = feature_names[stable_idx],
    stability = stability_scores[stable_idx]
  )

  stable_features <- stable_features[order(-stable_features$stability), ]

  list(
    selected_features = stable_features,
    n_selected = nrow(stable_features),
    stability_scores = stability_scores,
    threshold = threshold,
    n_bootstrap = n_bootstrap
  )
}

#' mRMR (minimum Redundancy Maximum Relevance) feature selection
#' @param oe OmicsExperiment or matrix
#' @param outcome vector of outcomes
#' @param n_features integer, number of features to select
#' @return list with selected features
#' @export
feature_select_mrmr <- function(oe, outcome, n_features = 50) {

  if (!requireNamespace("mRMRe", quietly = TRUE)) {
    warning("mRMRe not available, using variance + correlation selection")
    return(.feature_select_variance_cor(oe, outcome, n_features))
  }

  # Extract data
  if (inherits(oe, "SummarizedExperiment")) {
    x <- t(SummarizedExperiment::assay(oe, 1))
    feature_names <- rownames(SummarizedExperiment::assay(oe, 1))
  } else {
    x <- t(as.matrix(oe))
    feature_names <- rownames(as.matrix(oe))
  }

  x[is.na(x)] <- 0

  # Discretize for mRMR
  x_discrete <- apply(x, 2, function(col) {
    cut(col, breaks = 5, labels = FALSE)
  })

  y_discrete <- if (is.numeric(outcome)) {
    cut(outcome, breaks = 5, labels = FALSE)
  } else {
    as.integer(as.factor(outcome))
  }

  # Combine
  data <- cbind(y_discrete, x_discrete)
  colnames(data) <- c("outcome", feature_names)

  # Run mRMR
  mrmr_data <- mRMRe::mRMR.data(data = data.frame(data))
  result <- mRMRe::mRMR.classic(
    data = mrmr_data,
    target_indices = 1,
    feature_count = min(n_features, ncol(x))
  )

  selected_idx <- mRMRe::solutions(result)[[1]]
  selected_features <- data.frame(
    feature = feature_names[selected_idx - 1],  # -1 for outcome column
    rank = 1:length(selected_idx)
  )

  list(
    selected_features = selected_features,
    n_selected = nrow(selected_features),
    mrmr_result = result
  )
}

#' Variance + correlation fallback
#' @keywords internal
.feature_select_variance_cor <- function(oe, outcome, n_features) {

  if (inherits(oe, "SummarizedExperiment")) {
    x <- t(SummarizedExperiment::assay(oe, 1))
    feature_names <- rownames(SummarizedExperiment::assay(oe, 1))
  } else {
    x <- t(as.matrix(oe))
    feature_names <- rownames(as.matrix(oe))
  }

  x[is.na(x)] <- 0

  # High variance features
  vars <- apply(x, 2, var)
  high_var_idx <- order(-vars)[1:min(n_features * 3, ncol(x))]

  # Among high variance, select by correlation
  x_subset <- x[, high_var_idx]
  y_numeric <- if (is.factor(outcome)) as.numeric(outcome) else outcome

  cors <- abs(apply(x_subset, 2, function(col) cor(col, y_numeric)))
  selected_idx <- high_var_idx[order(-cors)[1:n_features]]

  selected_features <- data.frame(
    feature = feature_names[selected_idx],
    variance = vars[selected_idx],
    abs_correlation = cors[order(-cors)[1:n_features]]
  )

  list(
    selected_features = selected_features,
    n_selected = n_features,
    method = "variance_cor_fallback"
  )
}

#' Combined feature selection
#' @param oe OmicsExperiment or matrix
#' @param outcome vector
#' @param methods character vector of methods to combine
#' @param voting_threshold numeric, fraction of methods that must select a feature
#' @return list with consensus features
#' @export
feature_select_consensus <- function(oe, outcome,
                                     methods = c("elastic_net", "boruta", "stability"),
                                     voting_threshold = 0.5) {

  all_selections <- list()

  if ("elastic_net" %in% methods) {
    result <- feature_select_elastic_net(oe, outcome)
    all_selections$elastic_net <- result$selected_features$feature
  }

  if ("boruta" %in% methods) {
    result <- feature_select_boruta(oe, outcome)
    all_selections$boruta <- result$selected_features$feature
  }

  if ("stability" %in% methods) {
    result <- feature_select_stability(oe, outcome)
    all_selections$stability <- result$selected_features$feature
  }

  # Count votes
  all_features <- unique(unlist(all_selections))
  votes <- sapply(all_features, function(f) {
    sum(sapply(all_selections, function(sel) f %in% sel))
  })

  vote_threshold <- ceiling(length(methods) * voting_threshold)
  consensus_features <- names(votes)[votes >= vote_threshold]

  data.frame(
    feature = consensus_features,
    votes = votes[consensus_features],
    vote_fraction = votes[consensus_features] / length(methods)
  )
}
