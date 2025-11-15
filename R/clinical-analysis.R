#' Clinical integration and outcomes analysis
#'
#' @description
#' Survival analysis, biomarker discovery, patient stratification

#' Survival analysis with omics features
#' @param clinical_project ClinicalOmicsProject
#' @param omics_assay character, name of omics assay to use
#' @param features character vector of feature names
#' @param formula optional survival formula
#' @return list with survival model results
#' @export
clinical_survival <- function(clinical_project, omics_assay = NULL,
                              features = NULL, formula = NULL) {

  if (!requireNamespace("survival", quietly = TRUE)) {
    stop("survival package required")
  }

  # Get survival data
  surv_data <- clinical_project@survival_data

  if (nrow(surv_data) == 0) {
    stop("No survival data available")
  }

  # Get omics data
  if (is.null(omics_assay)) {
    omics_assay <- names(MultiAssayExperiment::experiments(clinical_project))[1]
  }

  oe <- MultiAssayExperiment::experiments(clinical_project)[[omics_assay]]
  expr <- t(SummarizedExperiment::assay(oe, 1))

  # Select features
  if (is.null(features)) {
    # Use top variance features
    vars <- apply(expr, 2, var, na.rm = TRUE)
    top_idx <- order(-vars)[1:min(100, ncol(expr))]
    expr <- expr[, top_idx]
  } else {
    expr <- expr[, features, drop = FALSE]
  }

  expr[is.na(expr)] <- 0

  # Combine with survival data
  surv_df <- as.data.frame(surv_data)
  surv_df <- cbind(surv_df, expr)

  # Fit Cox models for each feature
  if (is.null(formula)) {
    results <- lapply(colnames(expr), function(feat) {
      form <- as.formula(paste0("Surv(time, event) ~ ", feat))

      fit <- survival::coxph(form, data = surv_df)
      summ <- summary(fit)

      data.frame(
        feature = feat,
        coef = summ$coefficients[1, 1],
        hr = summ$coefficients[1, 2],
        se = summ$coefficients[1, 3],
        z = summ$coefficients[1, 4],
        pval = summ$coefficients[1, 5],
        concordance = summ$concordance[1]
      )
    })

    results_df <- do.call(rbind, results)
    results_df$fdr <- p.adjust(results_df$pval, method = "BH")
    results_df <- results_df[order(results_df$pval), ]

  } else {
    # User-provided formula
    fit <- survival::coxph(formula, data = surv_df)
    results_df <- summary(fit)
  }

  list(
    results = results_df,
    surv_data = surv_data,
    n_patients = nrow(surv_data),
    n_events = sum(surv_data$event)
  )
}

#' Biomarker discovery
#' @param clinical_project ClinicalOmicsProject
#' @param outcome character or numeric vector
#' @param omics_assay character, omics assay name
#' @param method character, selection method
#' @return DataFrame of discovered biomarkers
#' @export
clinical_biomarkers <- function(clinical_project, outcome,
                                omics_assay = NULL, method = "elastic_net") {

  # Get omics data
  if (is.null(omics_assay)) {
    omics_assay <- names(MultiAssayExperiment::experiments(clinical_project))[1]
  }

  oe <- MultiAssayExperiment::experiments(clinical_project)[[omics_assay]]

  # Feature selection
  if (method == "elastic_net") {
    result <- feature_select_elastic_net(oe, outcome, alpha = 0.5)
    biomarkers <- result$selected_features

  } else if (method == "boruta") {
    result <- feature_select_boruta(oe, outcome)
    biomarkers <- result$selected_features[result$selected_features$decision == "Confirmed", ]

  } else if (method == "cox") {
    # Use survival analysis
    if (!is.numeric(outcome)) {
      stop("Cox method requires survival data")
    }

    surv_result <- clinical_survival(clinical_project, omics_assay = omics_assay)
    biomarkers <- surv_result$results[surv_result$results$fdr < 0.05, ]

  } else {
    stop("Unknown method: ", method)
  }

  # Store in project
  clinical_project@biomarkers <- S4Vectors::DataFrame(biomarkers)

  list(
    biomarkers = biomarkers,
    n_biomarkers = nrow(biomarkers),
    method = method,
    project = clinical_project
  )
}

#' Patient stratification
#' @param clinical_project ClinicalOmicsProject
#' @param omics_assay character
#' @param features character vector of features for stratification
#' @param n_groups integer, number of patient groups
#' @param method character, clustering method
#' @return updated project with stratification
#' @export
clinical_stratify <- function(clinical_project, omics_assay = NULL,
                              features = NULL, n_groups = 3,
                              method = "kmeans") {

  # Get omics data
  if (is.null(omics_assay)) {
    omics_assay <- names(MultiAssayExperiment::experiments(clinical_project))[1]
  }

  oe <- MultiAssayExperiment::experiments(clinical_project)[[omics_assay]]
  expr <- t(SummarizedExperiment::assay(oe, 1))

  # Select features
  if (!is.null(features)) {
    expr <- expr[, features, drop = FALSE]
  } else {
    # Use biomarkers if available
    if (nrow(clinical_project@biomarkers) > 0) {
      features <- clinical_project@biomarkers$feature
      expr <- expr[, features, drop = FALSE]
    } else {
      # Top variable features
      vars <- apply(expr, 2, var, na.rm = TRUE)
      top_idx <- order(-vars)[1:min(50, ncol(expr))]
      expr <- expr[, top_idx]
    }
  }

  expr[is.na(expr)] <- 0
  expr_scaled <- scale(expr)

  # Clustering
  if (method == "kmeans") {
    clusters <- kmeans(expr_scaled, centers = n_groups, nstart = 25)$cluster

  } else if (method == "hierarchical") {
    hc <- hclust(dist(expr_scaled), method = "ward.D2")
    clusters <- cutree(hc, k = n_groups)

  } else {
    stop("Unknown method: ", method)
  }

  # Assign to project
  clinical_project@stratification <- factor(clusters)

  # Validate with survival if available
  if (nrow(clinical_project@survival_data) > 0 &&
      requireNamespace("survival", quietly = TRUE)) {

    surv_data <- as.data.frame(clinical_project@survival_data)
    surv_data$group <- factor(clusters)

    fit <- survival::survdiff(Surv(time, event) ~ group, data = surv_data)
    pval <- 1 - pchisq(fit$chisq, df = length(fit$n) - 1)

    message("Stratification p-value (log-rank test): ", signif(pval, 3))
  }

  list(
    project = clinical_project,
    stratification = factor(clusters),
    n_groups = n_groups,
    method = method
  )
}

#' Clinical prediction model
#' @param clinical_project ClinicalOmicsProject
#' @param outcome vector, clinical outcome
#' @param omics_assay character
#' @param clinical_vars character vector of clinical variable names
#' @param method character, prediction method
#' @return list with model and performance
#' @export
clinical_predict <- function(clinical_project, outcome, omics_assay = NULL,
                             clinical_vars = NULL, method = "rf") {

  # Get omics data
  if (is.null(omics_assay)) {
    omics_assay <- names(MultiAssayExperiment::experiments(clinical_project))[1]
  }

  oe <- MultiAssayExperiment::experiments(clinical_project)[[omics_assay]]
  omics_data <- t(SummarizedExperiment::assay(oe, 1))

  # Get clinical data
  clinical_data <- as.data.frame(clinical_project@clinical_data)

  # Combine features
  if (!is.null(clinical_vars)) {
    combined_data <- cbind(
      clinical_data[, clinical_vars, drop = FALSE],
      omics_data
    )
  } else {
    combined_data <- omics_data
  }

  combined_data[is.na(combined_data)] <- 0

  # Train model
  if (method == "rf") {
    model <- ensemble_rf(t(combined_data), outcome, n_trees = 500)

  } else if (method == "xgboost") {
    model <- ensemble_xgboost(t(combined_data), outcome, n_rounds = 100)

  } else if (method == "glm") {
    df <- as.data.frame(combined_data)
    df$outcome <- outcome

    if (is.factor(outcome) || is.character(outcome)) {
      model <- glm(outcome ~ ., data = df, family = binomial())
    } else {
      model <- lm(outcome ~ ., data = df)
    }

  } else {
    stop("Unknown method: ", method)
  }

  # Cross-validation performance
  cv_results <- .cv_performance(combined_data, outcome, method = method)

  # Store predictions
  clinical_project@predictions[[paste0("model_", method)]] <- list(
    model = model,
    cv_performance = cv_results
  )

  list(
    model = model,
    cv_performance = cv_results,
    project = clinical_project,
    method = method
  )
}

#' Cross-validation performance
#' @keywords internal
.cv_performance <- function(x, y, method = "rf", n_folds = 5) {

  n <- nrow(x)
  folds <- sample(rep(1:n_folds, length.out = n))

  predictions <- numeric(n)

  for (fold in 1:n_folds) {
    train_idx <- which(folds != fold)
    test_idx <- which(folds == fold)

    train_x <- x[train_idx, ]
    train_y <- y[train_idx]
    test_x <- x[test_idx, ]

    # Train
    if (method == "rf") {
      model <- ensemble_rf(t(train_x), train_y, n_trees = 100, importance = FALSE)
      preds <- predict_ensemble(model, t(test_x))

    } else if (method == "xgboost") {
      model <- ensemble_xgboost(t(train_x), train_y, n_rounds = 50)
      preds <- predict_ensemble(model, t(test_x))

    } else {
      df_train <- as.data.frame(train_x)
      df_train$outcome <- train_y
      df_test <- as.data.frame(test_x)

      if (is.factor(train_y) || is.character(train_y)) {
        model <- glm(outcome ~ ., data = df_train, family = binomial())
        preds <- predict(model, newdata = df_test, type = "response")
      } else {
        model <- lm(outcome ~ ., data = df_train)
        preds <- predict(model, newdata = df_test)
      }
    }

    predictions[test_idx] <- preds
  }

  # Compute metrics
  if (is.factor(y) || is.character(y)) {
    # Classification
    if (requireNamespace("pROC", quietly = TRUE)) {
      roc_obj <- pROC::roc(y, predictions)
      auc <- as.numeric(pROC::auc(roc_obj))
    } else {
      auc <- NA
    }

    list(
      predictions = predictions,
      auc = auc,
      type = "classification"
    )

  } else {
    # Regression
    rmse <- sqrt(mean((y - predictions)^2))
    r2 <- cor(y, predictions)^2

    list(
      predictions = predictions,
      rmse = rmse,
      r2 = r2,
      type = "regression"
    )
  }
}

#' Plot survival curves by group
#' @param clinical_project ClinicalOmicsProject
#' @param groups factor of patient groups
#' @return survival plot
#' @export
plot_survival <- function(clinical_project, groups = NULL) {

  if (!requireNamespace("survival", quietly = TRUE)) {
    stop("survival package required")
  }

  surv_data <- as.data.frame(clinical_project@survival_data)

  if (is.null(groups)) {
    groups <- clinical_project@stratification
  }

  if (length(groups) == 0) {
    stop("No groups provided")
  }

  surv_data$group <- groups

  fit <- survival::survfit(Surv(time, event) ~ group, data = surv_data)

  # Plot
  if (requireNamespace("survminer", quietly = TRUE)) {
    survminer::ggsurvplot(
      fit,
      data = surv_data,
      pval = TRUE,
      conf.int = TRUE,
      risk.table = TRUE
    )
  } else {
    plot(fit, col = 1:length(unique(groups)), lwd = 2,
         xlab = "Time", ylab = "Survival Probability")
    legend("topright", legend = levels(groups), col = 1:length(unique(groups)), lwd = 2)
  }
}
