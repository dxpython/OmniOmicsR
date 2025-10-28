test_that("assoc_corr works with covariate", {
  f <- system.file("extdata/example_counts.csv", package = "OmniOmicsR")
  oe <- read_omics_matrix(f)
  cd <- SummarizedExperiment::colData(oe)
  cd$covar <- seq_len(nrow(cd))
  SummarizedExperiment::colData(oe) <- cd
  res <- assoc_corr(oe, "covar")
  expect_true(all(c("feature_x","feature_y","cor","pval") %in% names(res)))
})

test_that("sc_hvg returns features", {
  f <- system.file("extdata/example_counts.csv", package = "OmniOmicsR")
  oe <- read_omics_matrix(f)
  hvg <- sc_hvg(oe, top = 2)
  expect_equal(length(hvg), 2)
})

