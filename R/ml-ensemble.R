#' Ensemble methods for omics prediction
#'
#' @description
#' Random forests, gradient boosting, and stacking for classification/regression

#' Random forest for omics data
#' @param oe OmicsExperiment or matrix (features x samples)
#' @param outcome vector of outcomes (factor for classification, numeric for regression)
#' @param n_trees integer, number of trees
#' @param mtry integer, number of variables to try at each split
#' @param importance logical, compute feature importance
#' @param ... additional arguments
#' @return list with model and predictions
#' @export
ensemble_rf <- function(oe, outcome, n_trees = 500, mtry = NULL,
                        importance = TRUE, ...) {

  # Extract data
  if (inherits(oe, "SummarizedExperiment")) {
    x <- t(SummarizedExperiment::assay(oe, 1))
  } else {
    x <- t(as.matrix(oe))
  }

  # Handle missing values
  x[is.na(x)] <- 0

  # Convert to data.frame
  df <- as.data.frame(x)
  df$outcome <- outcome

  # Determine if classification or regression
  is_classification <- is.factor(outcome) || is.character(outcome)

  if (is.null(mtry)) {
    mtry <- if (is_classification) floor(sqrt(ncol(x))) else floor(ncol(x) / 3)
  }

  # Use ranger if available, otherwise randomForest
  if (requireNamespace("ranger", quietly = TRUE)) {
    model <- ranger::ranger(
      outcome ~ .,
      data = df,
      num.trees = n_trees,
      mtry = mtry,
      importance = if (importance) "impurity" else "none",
      probability = is_classification,
      ...
    )

    imp <- if (importance) {
      data.frame(
        feature = names(ranger::importance(model)),
        importance = as.numeric(ranger::importance(model))
      )
    } else {
      NULL
    }

  } else if (requireNamespace("randomForest", quietly = TRUE)) {
    model <- randomForest::randomForest(
      outcome ~ .,
      data = df,
      ntree = n_trees,
      mtry = mtry,
      importance = importance,
      ...
    )

    imp <- if (importance) {
      imp_mat <- randomForest::importance(model)
      data.frame(
        feature = rownames(imp_mat),
        importance = imp_mat[, 1]
      )
    } else {
      NULL
    }

  } else {
    stop("Neither ranger nor randomForest package is available")
  }

  # Order by importance
  if (!is.null(imp)) {
    imp <- imp[order(-imp$importance), ]
  }

  list(
    model = model,
    importance = imp,
    mtry = mtry,
    n_trees = n_trees,
    type = if (is_classification) "classification" else "regression"
  )
}

#' Gradient boosting for omics data
#' @param oe OmicsExperiment or matrix
#' @param outcome vector of outcomes
#' @param n_rounds integer, boosting rounds
#' @param max_depth integer, tree depth
#' @param eta numeric, learning rate
#' @param ... additional xgboost parameters
#' @return list with model and predictions
#' @export
ensemble_xgboost <- function(oe, outcome, n_rounds = 100, max_depth = 6,
                              eta = 0.3, ...) {

  if (!requireNamespace("xgboost", quietly = TRUE)) {
    stop("xgboost package required")
  }

  # Extract data
  if (inherits(oe, "SummarizedExperiment")) {
    x <- t(SummarizedExperiment::assay(oe, 1))
  } else {
    x <- t(as.matrix(oe))
  }

  x[is.na(x)] <- 0

  # Determine task type
  is_classification <- is.factor(outcome) || is.character(outcome)

  if (is_classification) {
    outcome <- as.integer(as.factor(outcome)) - 1
    num_class <- length(unique(outcome))
    objective <- if (num_class == 2) "binary:logistic" else "multi:softmax"
  } else {
    num_class <- NULL
    objective <- "reg:squarederror"
  }

  # Create DMatrix
  dtrain <- xgboost::xgb.DMatrix(data = x, label = outcome)

  # Train
  params <- list(
    objective = objective,
    max_depth = max_depth,
    eta = eta,
    ...
  )

  if (!is.null(num_class) && num_class > 2) {
    params$num_class <- num_class
  }

  model <- xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = n_rounds,
    verbose = 0
  )

  # Feature importance
  imp <- xgboost::xgb.importance(model = model)

  list(
    model = model,
    importance = imp,
    params = params,
    n_rounds = n_rounds,
    type = if (is_classification) "classification" else "regression"
  )
}

#' Ensemble stacking
#' @param oe OmicsExperiment or matrix
#' @param outcome vector of outcomes
#' @param base_learners list of functions to use as base learners
#' @param meta_learner function for meta-learning
#' @param cv_folds integer, cross-validation folds
#' @return list with stacked model
#' @export
ensemble_stack <- function(oe, outcome, base_learners = NULL,
                           meta_learner = NULL, cv_folds = 5) {

  # Extract data
  if (inherits(oe, "SummarizedExperiment")) {
    x <- t(SummarizedExperiment::assay(oe, 1))
  } else {
    x <- t(as.matrix(oe))
  }

  x[is.na(x)] <- 0
  n_samples <- nrow(x)

  # Default base learners
  if (is.null(base_learners)) {
    base_learners <- list(
      rf = function(train_x, train_y) {
        ensemble_rf(t(train_x), train_y, n_trees = 100, importance = FALSE)
      },
      xgb = function(train_x, train_y) {
        ensemble_xgboost(t(train_x), train_y, n_rounds = 50)
      }
    )
  }

  # Create CV folds
  folds <- sample(rep(1:cv_folds, length.out = n_samples))

  # Generate meta-features
  meta_features <- matrix(NA, nrow = n_samples, ncol = length(base_learners))
  colnames(meta_features) <- names(base_learners)

  for (fold in 1:cv_folds) {
    train_idx <- which(folds != fold)
    test_idx <- which(folds == fold)

    train_x <- x[train_idx, , drop = FALSE]
    train_y <- outcome[train_idx]
    test_x <- x[test_idx, , drop = FALSE]

    # Train each base learner
    for (i in seq_along(base_learners)) {
      model <- base_learners[[i]](train_x, train_y)

      # Get predictions
      if (inherits(model$model, "ranger")) {
        preds <- predict(model$model, data.frame(test_x))$predictions
      } else if (inherits(model$model, "randomForest")) {
        preds <- predict(model$model, newdata = data.frame(test_x))
      } else if (inherits(model$model, "xgb.Booster")) {
        dtest <- xgboost::xgb.DMatrix(data = test_x)
        preds <- predict(model$model, dtest)
      }

      meta_features[test_idx, i] <- preds
    }
  }

  # Train meta-learner on full data
  full_models <- lapply(base_learners, function(f) f(x, outcome))

  # Train meta-learner
  if (is.null(meta_learner)) {
    # Default: GLM for regression, GLM with family for classification
    if (is.factor(outcome) || is.character(outcome)) {
      meta_model <- glm(outcome ~ ., data = data.frame(meta_features, outcome = outcome),
                        family = binomial())
    } else {
      meta_model <- lm(outcome ~ ., data = data.frame(meta_features, outcome = outcome))
    }
  } else {
    meta_model <- meta_learner(meta_features, outcome)
  }

  list(
    base_models = full_models,
    meta_model = meta_model,
    meta_features = meta_features,
    cv_folds = cv_folds
  )
}

#' Predict with ensemble model
#' @param ensemble_result result from ensemble_rf, ensemble_xgboost, or ensemble_stack
#' @param new_data new data matrix or OmicsExperiment
#' @return vector of predictions
#' @export
predict_ensemble <- function(ensemble_result, new_data) {

  # Extract data
  if (inherits(new_data, "SummarizedExperiment")) {
    x <- t(SummarizedExperiment::assay(new_data, 1))
  } else {
    x <- t(as.matrix(new_data))
  }

  x[is.na(x)] <- 0

  # Check if stacked model
  if ("base_models" %in% names(ensemble_result)) {
    # Stacked model: get predictions from base models
    base_preds <- sapply(ensemble_result$base_models, function(model) {
      predict_ensemble(model, t(x))
    })

    # Meta-learner prediction
    preds <- predict(ensemble_result$meta_model, newdata = data.frame(base_preds))

  } else {
    # Single model
    model <- ensemble_result$model

    if (inherits(model, "ranger")) {
      preds <- predict(model, data = data.frame(x))$predictions
    } else if (inherits(model, "randomForest")) {
      preds <- predict(model, newdata = data.frame(x))
    } else if (inherits(model, "xgb.Booster")) {
      dtest <- xgboost::xgb.DMatrix(data = x)
      preds <- predict(model, dtest)
    }
  }

  preds
}

#' Plot feature importance from ensemble
#' @param ensemble_result result from ensemble function
#' @param top_n integer, number of top features to plot
#' @return ggplot2 object
#' @export
plot_ensemble_importance <- function(ensemble_result, top_n = 20) {

  if (is.null(ensemble_result$importance)) {
    stop("No importance information available")
  }

  imp <- ensemble_result$importance
  imp <- head(imp[order(-imp$importance), ], top_n)

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    imp$feature <- factor(imp$feature, levels = rev(imp$feature))

    ggplot2::ggplot(imp, ggplot2::aes(x = importance, y = feature)) +
      ggplot2::geom_col(fill = "steelblue") +
      ggplot2::labs(
        title = "Feature Importance",
        x = "Importance",
        y = "Feature"
      ) +
      ggplot2::theme_minimal()
  } else {
    # Base R plot
    par(mar = c(4, 8, 2, 1))
    barplot(rev(imp$importance), names.arg = rev(imp$feature),
            horiz = TRUE, las = 1,
            main = "Feature Importance",
            xlab = "Importance")
  }
}
