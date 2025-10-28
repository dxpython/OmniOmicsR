test_that("read_omics_matrix constructs OmicsExperiment", {
  f <- system.file("extdata/example_counts.csv", package = "OmniOmicsR")
  expect_true(file.exists(f))
  oe <- read_omics_matrix(f, omics_type = "rna")
  expect_s4_class(oe, "OmicsExperiment")
  expect_true(nrow(oe) > 0)
  expect_true(ncol(oe) > 0)
})

test_that("read_maxquant parses intensities and filters contaminants", {
  f <- system.file("extdata/proteinGroups_min.txt", package = "OmniOmicsR")
  expect_true(file.exists(f))
  oe <- read_maxquant(f)
  expect_s4_class(oe, "OmicsExperiment")
  # contaminants removed: expect 2 rows (P001, P002)
  expect_equal(nrow(oe), 2)
  expect_setequal(colnames(oe), c("S1", "S2"))
})

test_that("read_mztab detects abundance columns", {
  f <- system.file("extdata/metabolites_mztab_min.tsv", package = "OmniOmicsR")
  expect_true(file.exists(f))
  oe <- read_mztab(f)
  expect_s4_class(oe, "OmicsExperiment")
  expect_equal(ncol(oe), 2)
})

test_that("read_seurat gracefully errors when Seurat missing", {
  skip_if(requireNamespace("Seurat", quietly = TRUE), "Seurat installed; skipping missing test")
  expect_error(read_seurat("nonexistent.rds"))
})
