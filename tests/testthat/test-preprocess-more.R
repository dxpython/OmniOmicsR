test_that("normalize_quantile creates qnorm assay (fallback ok)", {
  f <- system.file("extdata/example_counts.csv", package = "OmniOmicsR")
  oe <- read_omics_matrix(f)
  oe2 <- normalize_quantile(oe)
  expect_true("qnorm" %in% names(SummarizedExperiment::assays(oe2)))
})

test_that("impute_knn produces imputed_knn assay (fallback ok)", {
  f <- system.file("extdata/example_counts.csv", package = "OmniOmicsR")
  oe <- read_omics_matrix(f)
  # inject some NAs
  m <- SummarizedExperiment::assay(oe, 1)
  m[1, 1] <- NA_real_
  SummarizedExperiment::assays(oe)[[1]] <- m
  oe2 <- impute_knn(oe)
  expect_true("imputed_knn" %in% names(SummarizedExperiment::assays(oe2)))
})

