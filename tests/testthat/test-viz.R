test_that("plot_volcano returns ggplot", {
  res <- data.frame(logFC = c(-2, 0, 1.5), `adj.P.Val` = c(0.01, 0.5, 0.03), feature = c("A","B","C"))
  p <- plot_volcano(res, lfc_col = "logFC", p_col = "adj.P.Val")
  expect_s3_class(p, "ggplot")
})

test_that("plot_umap skips if uwot missing", {
  skip_if(requireNamespace("uwot", quietly = TRUE), "uwot installed; skipping negative path")
  f <- system.file("extdata/example_counts.csv", package = "OmniOmicsR")
  oe <- read_omics_matrix(f)
  expect_error(plot_umap(oe))
})

