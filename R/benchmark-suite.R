#' Comprehensive benchmarking suite
#'
#' @description
#' Performance testing for all major functions

#' Benchmark all major functions
#' @param n_features integer, number of features
#' @param n_samples integer, number of samples
#' @param verbose logical
#' @return data.frame with benchmark results
#' @export
benchmark_all <- function(n_features = 10000, n_samples = 1000, verbose = TRUE) {

  message("========================================")
  message("OmniOmicsR v2.0 Comprehensive Benchmark")
  message("Features: ", n_features, " | Samples: ", n_samples)
  message("========================================\n")

  results <- list()

  # 1. Data simulation
  if (verbose) message("[1/10] Benchmarking data simulation...")
  results$simulation <- system.time({
    rna_data <- simulate_rnaseq(n_features, n_samples)
  })

  # 2. Normalization
  if (verbose) message("[2/10] Benchmarking normalization methods...")
  results$normalization_tmm <- system.time({
    norm_tmm <- normalize_tmm(rna_data)
  })

  results$normalization_vst <- system.time({
    norm_vst <- normalize_vst(rna_data)
  })

  # 3. Feature selection
  if (verbose) message("[3/10] Benchmarking feature selection...")
  outcome <- rep(c(0, 1), length.out = n_samples)

  results$feature_elastic_net <- system.time({
    fs_en <- feature_select_elastic_net(rna_data, outcome)
  })

  # 4. Machine learning - Random Forest
  if (verbose) message("[4/10] Benchmarking Random Forest...")
  results$ml_rf <- system.time({
    rf_model <- ensemble_rf(rna_data, outcome, n_trees = 100, importance = TRUE)
  })

  # 5. VAE
  if (verbose) message("[5/10] Benchmarking VAE (using PCA fallback)...")
  results$ml_vae <- system.time({
    vae_result <- train_vae(rna_data, latent_dim = 32, epochs = 10)
  })

  # 6. Differential expression
  if (verbose) message("[6/10] Benchmarking differential expression...")
  results$dea_deseq2 <- system.time({
    dea_result <- dea_deseq2(rna_data, design = ~group)
  })

  # 7. Network analysis
  if (verbose) message("[7/10] Benchmarking network analysis...")
  # Use smaller subset for network analysis
  subset_data <- rna_data[1:min(500, n_features), 1:min(200, n_samples)]

  results$network_wgcna <- system.time({
    net_result <- network_wgcna(subset_data, power = 6, min_module_size = 10)
  })

  # 8. Spatial omics
  if (verbose) message("[8/10] Benchmarking spatial omics...")
  results$spatial_simulation <- system.time({
    spatial_data <- simulate_spatial(n_features = 1000, n_spots = min(500, n_samples))
  })

  results$spatial_clustering <- system.time({
    spatial_clusters <- spatial_clustering(spatial_data, n_clusters = 5)
  })

  # 9. Single-cell multi-omics
  if (verbose) message("[9/10] Benchmarking sc multi-omics...")
  results$sc_multiomics_simulation <- system.time({
    sc_data <- simulate_sc_multiomics(n_genes = 1000, n_peaks = 2000,
                                      n_cells = min(500, n_samples))
  })

  results$sc_integration <- system.time({
    sc_integrated <- sc_integrate_modalities(sc_data, k = 10)
  })

  # 10. Multi-omics integration
  if (verbose) message("[10/10] Benchmarking multi-omics integration...")
  protein_data <- simulate_proteomics(n_features = 1000, n_samples = min(200, n_samples))

  results$integration_diablo <- system.time({
    int_result <- integrate_diablo(list(RNA = rna_data[1:1000, 1:min(200, n_samples)],
                                        Protein = protein_data),
                                  outcome = outcome[1:min(200, n_samples)])
  })

  # Compile results
  benchmark_df <- data.frame(
    function_name = names(results),
    user_time = sapply(results, function(x) x["user.self"]),
    system_time = sapply(results, function(x) x["sys.self"]),
    elapsed_time = sapply(results, function(x) x["elapsed"]),
    row.names = NULL
  )

  benchmark_df$total_time <- benchmark_df$user_time + benchmark_df$system_time

  message("\n========================================")
  message("Benchmark Complete!")
  message("Total elapsed time: ", round(sum(benchmark_df$elapsed_time), 2), " seconds")
  message("========================================\n")

  benchmark_df
}

#' Memory profiling for large datasets
#' @param n_features integer
#' @param n_samples integer
#' @return data.frame with memory usage
#' @export
benchmark_memory <- function(n_features = 10000, n_samples = 1000) {

  message("Memory profiling for ", n_features, "x", n_samples, " dataset...")

  results <- list()

  # Baseline
  gc()
  baseline_mem <- sum(gc()[, 2])

  # Simulate data
  gc()
  rna_data <- simulate_rnaseq(n_features, n_samples)
  gc()
  after_sim <- sum(gc()[, 2])
  results$simulation <- after_sim - baseline_mem

  # Normalization
  gc()
  norm_data <- normalize_tmm(rna_data)
  gc()
  after_norm <- sum(gc()[, 2])
  results$normalization <- after_norm - after_sim

  # Feature selection
  gc()
  outcome <- rep(c(0, 1), length.out = n_samples)
  fs_result <- feature_select_elastic_net(rna_data, outcome)
  gc()
  after_fs <- sum(gc()[, 2])
  results$feature_selection <- after_fs - after_norm

  # ML model
  gc()
  rf_model <- ensemble_rf(rna_data, outcome, n_trees = 100)
  gc()
  after_ml <- sum(gc()[, 2])
  results$machine_learning <- after_ml - after_fs

  data.frame(
    component = names(results),
    memory_mb = sapply(results, function(x) x),
    row.names = NULL
  )
}

#' Scalability test - vary dataset size
#' @param feature_sizes integer vector of feature counts to test
#' @param sample_sizes integer vector of sample counts to test
#' @param n_reps integer, number of replicates
#' @return data.frame with scalability results
#' @export
benchmark_scalability <- function(feature_sizes = c(1000, 5000, 10000),
                                  sample_sizes = c(100, 500, 1000),
                                  n_reps = 3) {

  message("Scalability benchmark...")
  message("Feature sizes: ", paste(feature_sizes, collapse = ", "))
  message("Sample sizes: ", paste(sample_sizes, collapse = ", "))
  message("Replicates: ", n_reps, "\n")

  results <- list()

  for (nf in feature_sizes) {
    for (ns in sample_sizes) {
      key <- paste0("F", nf, "_S", ns)
      message("Testing ", key, "...")

      times <- numeric(n_reps)

      for (rep in 1:n_reps) {
        time <- system.time({
          # Simulate
          data <- simulate_rnaseq(n_features = nf, n_samples = ns)

          # Normalize
          norm <- normalize_tmm(data)

          # Feature selection
          outcome <- rep(c(0, 1), length.out = ns)
          fs <- feature_select_elastic_net(data, outcome)
        })

        times[rep] <- time["elapsed"]
      }

      results[[key]] <- data.frame(
        n_features = nf,
        n_samples = ns,
        mean_time = mean(times),
        sd_time = sd(times),
        min_time = min(times),
        max_time = max(times)
      )
    }
  }

  do.call(rbind, results)
}

#' Parallel processing benchmark
#' @param n_features integer
#' @param n_samples integer
#' @param n_cores integer, number of cores to test
#' @return data.frame with parallel performance
#' @export
benchmark_parallel <- function(n_features = 10000, n_samples = 1000, n_cores = c(1, 2, 4)) {

  message("Parallel processing benchmark...")

  if (!requireNamespace("future", quietly = TRUE) ||
      !requireNamespace("furrr", quietly = TRUE)) {
    message("future/furrr packages not available, skipping parallel benchmark")
    return(data.frame())
  }

  results <- list()

  # Simulate data once
  rna_data <- simulate_rnaseq(n_features, n_samples)
  outcome <- rep(c(0, 1), length.out = n_samples)

  for (nc in n_cores) {
    message("Testing with ", nc, " cores...")

    # Set up parallel backend
    future::plan(future::multicore, workers = nc)

    time <- system.time({
      # Parallel feature selection on subsets
      n_chunks <- 10
      chunk_size <- ceiling(n_features / n_chunks)

      results_parallel <- furrr::future_map(1:n_chunks, function(i) {
        start_idx <- (i - 1) * chunk_size + 1
        end_idx <- min(i * chunk_size, n_features)

        subset_data <- rna_data[start_idx:end_idx, ]
        feature_select_elastic_net(subset_data, outcome)
      })
    })

    results[[paste0("cores_", nc)]] <- data.frame(
      n_cores = nc,
      elapsed_time = time["elapsed"]
    )

    # Reset to sequential
    future::plan(future::sequential)
  }

  do.call(rbind, results)
}

#' Compare to baseline (original OmniOmicsR)
#' @param n_features integer
#' @param n_samples integer
#' @return comparison data.frame
#' @export
benchmark_vs_baseline <- function(n_features = 5000, n_samples = 500) {

  message("Benchmarking new features vs baseline...")

  results <- list()

  # Simulate data
  rna_data <- simulate_rnaseq(n_features, n_samples)
  outcome <- rep(c(0, 1), length.out = n_samples)

  # Baseline methods (from original package)
  results$baseline_normalization <- system.time({
    norm <- normalize_tmm(rna_data)
  })

  results$baseline_dea <- system.time({
    dea <- dea_deseq2(rna_data, design = ~group)
  })

  # New advanced methods
  results$advanced_vae <- system.time({
    vae <- train_vae(rna_data, latent_dim = 32, epochs = 10)
  })

  results$advanced_ensemble <- system.time({
    rf <- ensemble_rf(rna_data, outcome, n_trees = 100)
  })

  results$advanced_feature_selection <- system.time({
    fs <- feature_select_elastic_net(rna_data, outcome)
  })

  data.frame(
    method = names(results),
    elapsed_time = sapply(results, function(x) x["elapsed"]),
    category = c("baseline", "baseline", "advanced", "advanced", "advanced"),
    row.names = NULL
  )
}

#' Generate benchmark report
#' @param benchmark_results data.frame from benchmark_all
#' @param output_file character, path to save report
#' @return invisible
#' @export
generate_benchmark_report <- function(benchmark_results, output_file = "benchmark_report.txt") {

  sink(output_file)

  cat("========================================\n")
  cat("OmniOmicsR v2.0 Benchmark Report\n")
  cat("Generated:", Sys.time(), "\n")
  cat("========================================\n\n")

  cat("System Information:\n")
  cat("R Version:", R.version.string, "\n")
  cat("Platform:", R.version$platform, "\n")
  cat("Cores:", parallel::detectCores(), "\n\n")

  cat("Benchmark Results:\n")
  cat("------------------\n")
  print(benchmark_results)

  cat("\n\nSummary Statistics:\n")
  cat("Total Elapsed Time:", round(sum(benchmark_results$elapsed_time), 2), "seconds\n")
  cat("Mean Time per Function:", round(mean(benchmark_results$elapsed_time), 2), "seconds\n")
  cat("Fastest Function:", benchmark_results$function_name[which.min(benchmark_results$elapsed_time)],
      "(", round(min(benchmark_results$elapsed_time), 2), "s)\n")
  cat("Slowest Function:", benchmark_results$function_name[which.max(benchmark_results$elapsed_time)],
      "(", round(max(benchmark_results$elapsed_time), 2), "s)\n")

  cat("\n========================================\n")

  sink()

  message("Benchmark report saved to: ", output_file)
  invisible()
}
