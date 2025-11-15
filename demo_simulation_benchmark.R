#!/usr/bin/env Rscript
#' Comprehensive Demo and Benchmark Script for OmniOmicsR v2.0
#'
#' This script demonstrates all major features with large-scale data (10K x 1K)
#' and runs comprehensive benchmarks

library(methods)
library(stats)
library(utils)

# Load required Bioconductor packages
suppressPackageStartupMessages({
  library(SummarizedExperiment)
  library(S4Vectors)
  library(MultiAssayExperiment)
})

# Set seed for reproducibility
set.seed(42)

cat("========================================\n")
cat("OmniOmicsR v2.0 - Comprehensive Demo\n")
cat("========================================\n\n")

# Source all R files in correct order (base classes first)
cat("[1/15] Loading OmniOmicsR v2.0...\n")

# Load in specific order to ensure dependencies
ordered_files <- c(
  "R/classes-OmicsExperiment.R",
  "R/classes-OmniProject.R",
  "R/classes-Enhanced.R"
)

# Source ordered files first
for (f in ordered_files) {
  if (file.exists(f)) source(f)
}

# Then source remaining files
r_files <- list.files("R", pattern = "\\.R$", full.names = TRUE)
r_files <- setdiff(r_files, ordered_files)
for (f in r_files) {
  source(f)
}

cat("Loaded", length(ordered_files) + length(r_files), "R modules\n\n")

# Parameters for large-scale simulation
N_FEATURES <- 10000
N_SAMPLES <- 1000

cat("========================================\n")
cat("PART 1: DATA SIMULATION\n")
cat("========================================\n\n")

# 1. RNA-seq data
cat("[2/15] Simulating RNA-seq data (10K x 1K)...\n")
start_time <- Sys.time()
rna_data <- simulate_rnaseq(
  n_features = N_FEATURES,
  n_samples = N_SAMPLES,
  n_groups = 2,
  de_fraction = 0.1,
  fold_change = 2
)
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat("✓ RNA-seq simulation complete:", round(elapsed, 2), "seconds\n")
cat("  Dimensions:", nrow(rna_data), "features x", ncol(rna_data), "samples\n\n")

# 2. Proteomics data
cat("[3/15] Simulating proteomics data (5K x 1K)...\n")
start_time <- Sys.time()
protein_data <- simulate_proteomics(
  n_features = 5000,
  n_samples = N_SAMPLES,
  n_groups = 2,
  missing_fraction = 0.2
)
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat("✓ Proteomics simulation complete:", round(elapsed, 2), "seconds\n\n")

# 3. Spatial transcriptomics
cat("[4/15] Simulating spatial transcriptomics...\n")
start_time <- Sys.time()
spatial_data <- simulate_spatial(
  n_features = 2000,
  n_spots = 500,
  grid_size = 50,
  n_regions = 5
)
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat("✓ Spatial data simulation complete:", round(elapsed, 2), "seconds\n\n")

# 4. Single-cell multi-omics
cat("[5/15] Simulating single-cell multi-omics...\n")
start_time <- Sys.time()
sc_data <- simulate_sc_multiomics(
  n_genes = 3000,
  n_peaks = 5000,
  n_cells = 1000,
  n_cell_types = 5
)
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat("✓ SC multi-omics simulation complete:", round(elapsed, 2), "seconds\n\n")

cat("========================================\n")
cat("PART 2: ADVANCED MACHINE LEARNING\n")
cat("========================================\n\n")

# Create outcome variable
outcome <- rep(c(0, 1), length.out = N_SAMPLES)

# 5. Feature selection
cat("[6/15] Running advanced feature selection...\n")
start_time <- Sys.time()
fs_result <- feature_select_elastic_net(rna_data, outcome, alpha = 0.5)
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat("✓ Feature selection complete:", round(elapsed, 2), "seconds\n")
cat("  Selected features:", nrow(fs_result$selected_features), "\n\n")

# 6. Random Forest
cat("[7/15] Training Random Forest ensemble...\n")
start_time <- Sys.time()
rf_model <- ensemble_rf(rna_data, outcome, n_trees = 200, importance = TRUE)
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat("✓ Random Forest training complete:", round(elapsed, 2), "seconds\n")
if (!is.null(rf_model$importance)) {
  cat("  Top 5 features by importance:\n")
  print(head(rf_model$importance, 5))
}
cat("\n")

# 7. VAE
cat("[8/15] Training Variational Autoencoder...\n")
start_time <- Sys.time()
vae_result <- train_vae(rna_data, latent_dim = 32, epochs = 10)
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat("✓ VAE training complete:", round(elapsed, 2), "seconds\n")
cat("  Latent dimensions:", ncol(vae_result$latent), "\n\n")

cat("========================================\n")
cat("PART 3: STATISTICAL ANALYSIS\n")
cat("========================================\n\n")

# 8. Differential expression
cat("[9/15] Running differential expression analysis...\n")
start_time <- Sys.time()
dea_result <- dea_deseq2(rna_data, design = ~group)
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat("✓ DEA complete:", round(elapsed, 2), "seconds\n")
cat("  Significant genes (FDR < 0.05):", sum(dea_result$padj < 0.05, na.rm = TRUE), "\n\n")

# 9. Network analysis (on subset)
cat("[10/15] Running WGCNA network analysis...\n")
start_time <- Sys.time()
subset_rna <- rna_data[1:500, 1:200]  # Subset for speed
network_result <- network_wgcna(subset_rna, power = 6, min_module_size = 10)
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat("✓ Network analysis complete:", round(elapsed, 2), "seconds\n")
cat("  Modules identified:", network_result$n_modules, "\n\n")

cat("========================================\n")
cat("PART 4: SPATIAL & SINGLE-CELL ANALYSIS\n")
cat("========================================\n\n")

# 10. Spatial clustering
cat("[11/15] Spatial clustering...\n")
start_time <- Sys.time()
spatial_clusters <- spatial_clustering(spatial_data, n_clusters = 5, method = "kmeans")
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat("✓ Spatial clustering complete:", round(elapsed, 2), "seconds\n")
cat("  Clusters:", table(spatial_clusters), "\n\n")

# 11. SC multi-omics integration
cat("[12/15] SC multi-omics integration...\n")
start_time <- Sys.time()
sc_integrated <- sc_integrate_modalities(sc_data, k = 20)
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat("✓ SC integration complete:", round(elapsed, 2), "seconds\n")
cat("  Integrated dimensions:", ncol(sc_integrated@integrated_embedding), "\n\n")

cat("========================================\n")
cat("PART 5: CLINICAL INTEGRATION\n")
cat("========================================\n\n")

# 12. Clinical project
cat("[13/15] Creating clinical omics project...\n")
start_time <- Sys.time()
clinical_project <- simulate_clinical_project(
  n_features = 5000,
  n_patients = 500,
  n_omics = 2,
  survival = TRUE
)
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat("✓ Clinical project created:", round(elapsed, 2), "seconds\n")
cat("  Patients:", nrow(clinical_project@clinical_data), "\n")
cat("  Omics layers:", length(MultiAssayExperiment::experiments(clinical_project)), "\n\n")

cat("========================================\n")
cat("PART 6: COMPREHENSIVE BENCHMARKING\n")
cat("========================================\n\n")

# 13. Full benchmark
cat("[14/15] Running comprehensive benchmark suite...\n")
cat("  (This may take several minutes)\n\n")
benchmark_results <- benchmark_all(n_features = N_FEATURES, n_samples = N_SAMPLES, verbose = TRUE)

cat("\n========================================\n")
cat("BENCHMARK RESULTS SUMMARY\n")
cat("========================================\n\n")
print(benchmark_results)

cat("\n========================================\n")
cat("SCALABILITY TEST\n")
cat("========================================\n\n")

# 14. Scalability benchmark
cat("[15/15] Running scalability tests...\n")
scalability_results <- benchmark_scalability(
  feature_sizes = c(1000, 5000, 10000),
  sample_sizes = c(100, 500, 1000),
  n_reps = 2
)

cat("\nScalability Results:\n")
print(scalability_results)

cat("\n========================================\n")
cat("MEMORY PROFILING\n")
cat("========================================\n\n")

memory_results <- benchmark_memory(n_features = N_FEATURES, n_samples = N_SAMPLES)
cat("Memory Usage by Component:\n")
print(memory_results)

cat("\n========================================\n")
cat("GENERATING BENCHMARK REPORT\n")
cat("========================================\n\n")

# Generate report
generate_benchmark_report(benchmark_results, "omniomicsr_v2_benchmark_report.txt")

cat("\n========================================\n")
cat("DEMONSTRATION COMPLETE!\n")
cat("========================================\n\n")

cat("Summary:\n")
cat("✓ Successfully simulated large-scale multi-omics data (10K x 1K)\n")
cat("✓ Demonstrated advanced ML features (VAE, ensemble, feature selection)\n")
cat("✓ Ran statistical analyses (DEA, network analysis)\n")
cat("✓ Tested spatial and single-cell multi-omics\n")
cat("✓ Integrated clinical data with omics\n")
cat("✓ Completed comprehensive benchmarking\n")
cat("✓ Tested scalability across multiple data sizes\n\n")

cat("Output files:\n")
cat("  - omniomicsr_v2_benchmark_report.txt\n\n")

cat("OmniOmicsR v2.0 is ready for production use!\n")
cat("========================================\n")

# Return benchmark results invisibly
invisible(list(
  benchmark_results = benchmark_results,
  scalability_results = scalability_results,
  memory_results = memory_results,
  simulated_data = list(
    rna = rna_data,
    protein = protein_data,
    spatial = spatial_data,
    sc_multiomics = sc_data,
    clinical = clinical_project
  ),
  analysis_results = list(
    feature_selection = fs_result,
    random_forest = rf_model,
    vae = vae_result,
    dea = dea_result,
    network = network_result,
    spatial_clusters = spatial_clusters,
    sc_integrated = sc_integrated
  )
))
