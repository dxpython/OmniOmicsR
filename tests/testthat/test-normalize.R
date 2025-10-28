test_that("normalize_tmm adds norm assay", {
  f <- system.file("extdata/example_counts.csv", package = "OmniOmicsR")
  oe <- read_omics_matrix(f)
  oe2 <- normalize_tmm(oe)
  expect_true("norm" %in% names(SummarizedExperiment::assays(oe2)))
})

