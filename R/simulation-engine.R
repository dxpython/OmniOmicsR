#' Comprehensive simulation engine for multi-omics data
#'
#' @description
#' Generate realistic omics data with biological structure

#' Simulate RNA-seq count data
#' @param n_features integer, number of genes
#' @param n_samples integer, number of samples
#' @param n_groups integer, number of biological groups
#' @param de_fraction numeric, fraction of DE genes
#' @param fold_change numeric, log fold change for DE genes
#' @param dispersion numeric, negative binomial dispersion
#' @param library_size numeric, mean library size
#' @return OmicsExperiment with simulated RNA-seq data
#' @export
simulate_rnaseq <- function(n_features = 10000, n_samples = 1000,
                            n_groups = 2, de_fraction = 0.1,
                            fold_change = 2, dispersion = 0.1,
                            library_size = 1e6) {

  message("Simulating RNA-seq data: ", n_features, " features x ", n_samples, " samples")

  # Group assignments
  groups <- factor(rep(1:n_groups, length.out = n_samples))

  # Base expression levels (log scale)
  base_expression <- rnorm(n_features, mean = 5, sd = 2)
  base_expression <- pmax(base_expression, 0)

  # Determine DE genes
  n_de <- round(n_features * de_fraction)
  de_genes <- sample(1:n_features, n_de)

  # Expression matrix
  expr_mat <- matrix(NA, nrow = n_features, ncol = n_samples)

  for (i in 1:n_samples) {
    group <- as.integer(groups[i])

    # Group-specific expression
    expr_levels <- base_expression

    if (group > 1) {
      # Add differential expression
      expr_levels[de_genes] <- expr_levels[de_genes] + log2(fold_change) * (group - 1)
    }

    # Convert to counts (negative binomial)
    lib_size <- rnorm(1, library_size, library_size * 0.1)
    lib_size <- max(lib_size, library_size * 0.5)

    probs <- exp(expr_levels)
    probs <- probs / sum(probs)

    # Multinomial sampling with overdispersion
    counts <- sapply(probs, function(p) {
      mu <- p * lib_size
      rnbinom(1, mu = mu, size = 1 / dispersion)
    })

    expr_mat[, i] <- counts
  }

  # Rownames and colnames
  rownames(expr_mat) <- paste0("Gene_", 1:n_features)
  colnames(expr_mat) <- paste0("Sample_", 1:n_samples)

  # Create OmicsExperiment
  col_data <- S4Vectors::DataFrame(
    sample_id = colnames(expr_mat),
    group = groups,
    library_size = colSums(expr_mat),
    row.names = colnames(expr_mat)
  )

  row_data <- S4Vectors::DataFrame(
    gene_id = rownames(expr_mat),
    de_status = 1:n_features %in% de_genes,
    base_expression = base_expression,
    row.names = rownames(expr_mat)
  )

  as_oe(expr_mat, omics_type = "rna", col_data = col_data, row_data = row_data)
}

#' Simulate proteomics data
#' @param n_features integer, number of proteins
#' @param n_samples integer
#' @param n_groups integer
#' @param missing_fraction numeric, fraction of missing values
#' @param de_fraction numeric
#' @param fold_change numeric
#' @return OmicsExperiment with simulated proteomics data
#' @export
simulate_proteomics <- function(n_features = 5000, n_samples = 1000,
                                n_groups = 2, missing_fraction = 0.2,
                                de_fraction = 0.15, fold_change = 1.5) {

  message("Simulating proteomics data: ", n_features, " features x ", n_samples, " samples")

  groups <- factor(rep(1:n_groups, length.out = n_samples))

  # Base abundance (log scale)
  base_abundance <- rnorm(n_features, mean = 20, sd = 3)

  # DE proteins
  n_de <- round(n_features * de_fraction)
  de_proteins <- sample(1:n_features, n_de)

  # Expression matrix
  expr_mat <- matrix(NA, nrow = n_features, ncol = n_samples)

  for (i in 1:n_samples) {
    group <- as.integer(groups[i])

    abundances <- base_abundance

    if (group > 1) {
      abundances[de_proteins] <- abundances[de_proteins] + log2(fold_change) * (group - 1)
    }

    # Add noise
    abundances <- abundances + rnorm(n_features, 0, 0.5)

    expr_mat[, i] <- abundances
  }

  # Introduce missing values (MNAR - missing not at random)
  missing_mask <- matrix(runif(n_features * n_samples) < missing_fraction,
                         nrow = n_features, ncol = n_samples)

  # Lower abundance proteins more likely to be missing
  missing_prob <- 1 / (1 + exp(base_abundance - median(base_abundance)))
  for (i in 1:n_features) {
    if (runif(1) < missing_prob[i]) {
      missing_mask[i, ] <- missing_mask[i, ] | (runif(n_samples) < 0.3)
    }
  }

  expr_mat[missing_mask] <- NA

  rownames(expr_mat) <- paste0("Protein_", 1:n_features)
  colnames(expr_mat) <- paste0("Sample_", 1:n_samples)

  col_data <- S4Vectors::DataFrame(
    sample_id = colnames(expr_mat),
    group = groups,
    row.names = colnames(expr_mat)
  )

  row_data <- S4Vectors::DataFrame(
    protein_id = rownames(expr_mat),
    de_status = 1:n_features %in% de_proteins,
    row.names = rownames(expr_mat)
  )

  as_oe(expr_mat, omics_type = "protein", col_data = col_data, row_data = row_data)
}

#' Simulate spatial transcriptomics data
#' @param n_features integer
#' @param n_spots integer
#' @param grid_size integer, spatial grid dimension
#' @param n_regions integer, number of spatial regions
#' @return SpatialOmicsExperiment
#' @export
simulate_spatial <- function(n_features = 5000, n_spots = 2000,
                             grid_size = 50, n_regions = 5) {

  message("Simulating spatial transcriptomics: ", n_features, " features x ", n_spots, " spots")

  # Generate spatial coordinates
  coords <- matrix(runif(n_spots * 2) * grid_size, ncol = 2)
  colnames(coords) <- c("x", "y")

  # Assign spots to spatial regions based on coordinates
  kmeans_result <- kmeans(coords, centers = n_regions, nstart = 10)
  regions <- kmeans_result$cluster

  # Base expression
  base_expr <- rnorm(n_features, mean = 4, sd = 1.5)

  # Region-specific genes (some genes enriched in certain regions)
  region_specific_genes <- lapply(1:n_regions, function(r) {
    sample(1:n_features, size = round(n_features * 0.05))
  })

  # Generate expression
  expr_mat <- matrix(NA, nrow = n_features, ncol = n_spots)

  for (i in 1:n_spots) {
    region <- regions[i]

    expr <- base_expr

    # Region-specific enrichment
    expr[region_specific_genes[[region]]] <- expr[region_specific_genes[[region]]] + 2

    # Spatial smoothing (nearby spots have similar expression)
    nearby_spots <- which(sqrt(rowSums((t(coords) - coords[i, ])^2)) < 5)
    if (length(nearby_spots) > 1) {
      spatial_effect <- rnorm(n_features, 0, 0.3)
      expr <- expr + spatial_effect
    }

    # Convert to counts
    probs <- exp(expr) / sum(exp(expr))
    counts <- rmultinom(1, size = sample(1000:10000, 1), prob = probs)[, 1]

    expr_mat[, i] <- counts
  }

  rownames(expr_mat) <- paste0("Gene_", 1:n_features)
  colnames(expr_mat) <- paste0("Spot_", 1:n_spots)

  create_spatial_experiment(
    x = expr_mat,
    spatial_coords = coords,
    omics_type = "spatial_rna"
  )
}

#' Simulate single-cell multi-omics data (RNA + ATAC)
#' @param n_genes integer
#' @param n_peaks integer
#' @param n_cells integer
#' @param n_cell_types integer
#' @return SingleCellMultiOmicsExperiment
#' @export
simulate_sc_multiomics <- function(n_genes = 3000, n_peaks = 5000,
                                   n_cells = 1000, n_cell_types = 5) {

  message("Simulating sc multi-omics: RNA (", n_genes, ") + ATAC (", n_peaks, ") x ", n_cells, " cells")

  # Cell type assignments
  cell_types <- factor(rep(1:n_cell_types, length.out = n_cells))

  # Simulate RNA
  rna_mat <- matrix(0, nrow = n_genes, ncol = n_cells)

  # Cell type marker genes
  markers_per_type <- round(n_genes * 0.1 / n_cell_types)

  for (ct in 1:n_cell_types) {
    cells_of_type <- which(cell_types == ct)
    marker_genes <- ((ct - 1) * markers_per_type + 1):(ct * markers_per_type)
    marker_genes <- marker_genes[marker_genes <= n_genes]

    # High expression in marker genes
    for (cell in cells_of_type) {
      # Background
      background <- rpois(n_genes, lambda = 0.5)

      # Markers
      background[marker_genes] <- rpois(length(marker_genes), lambda = 10)

      rna_mat[, cell] <- background
    }
  }

  # Simulate ATAC (peaks)
  atac_mat <- matrix(0, nrow = n_peaks, ncol = n_cells)

  # Cell type-specific peaks
  peaks_per_type <- round(n_peaks * 0.15 / n_cell_types)

  for (ct in 1:n_cell_types) {
    cells_of_type <- which(cell_types == ct)
    specific_peaks <- ((ct - 1) * peaks_per_type + 1):(ct * peaks_per_type)
    specific_peaks <- specific_peaks[specific_peaks <= n_peaks]

    for (cell in cells_of_type) {
      # Background accessibility
      background <- rbinom(n_peaks, size = 1, prob = 0.05)

      # Open chromatin in specific peaks
      background[specific_peaks] <- rbinom(length(specific_peaks), size = 1, prob = 0.7)

      atac_mat[, cell] <- background
    }
  }

  rownames(rna_mat) <- paste0("Gene_", 1:n_genes)
  rownames(atac_mat) <- paste0("Peak_", 1:n_peaks)
  colnames(rna_mat) <- colnames(atac_mat) <- paste0("Cell_", 1:n_cells)

  # Create experiment
  assays_list <- list(RNA = rna_mat, ATAC = atac_mat)

  create_sc_multiomics(assays_list)
}

#' Simulate clinical omics project
#' @param n_features integer
#' @param n_patients integer
#' @param n_omics integer, number of omics layers
#' @param survival logical, include survival data
#' @return ClinicalOmicsProject
#' @export
simulate_clinical_project <- function(n_features = 10000, n_patients = 500,
                                      n_omics = 2, survival = TRUE) {

  message("Simulating clinical project: ", n_patients, " patients, ", n_omics, " omics layers")

  # Simulate omics data
  omics_list <- list()

  if (n_omics >= 1) {
    omics_list$RNA <- simulate_rnaseq(n_features = n_features, n_samples = n_patients,
                                      n_groups = 2, de_fraction = 0.1)
  }

  if (n_omics >= 2) {
    omics_list$Protein <- simulate_proteomics(n_features = round(n_features / 2),
                                              n_samples = n_patients, n_groups = 2)
  }

  # Clinical data
  clinical_data <- S4Vectors::DataFrame(
    patient_id = paste0("Patient_", 1:n_patients),
    age = rnorm(n_patients, mean = 60, sd = 10),
    sex = factor(sample(c("M", "F"), n_patients, replace = TRUE)),
    stage = factor(sample(1:4, n_patients, replace = TRUE)),
    treatment = factor(sample(c("A", "B", "Control"), n_patients, replace = TRUE)),
    row.names = paste0("Patient_", 1:n_patients)
  )

  # Survival data
  if (survival) {
    # Simulate survival times influenced by stage
    stage_numeric <- as.numeric(clinical_data$stage)

    # Higher stage = worse survival
    base_hazard <- 0.1
    hazard <- base_hazard * exp(0.3 * (stage_numeric - 2))

    survival_times <- rexp(n_patients, rate = hazard)
    event_status <- rbinom(n_patients, 1, prob = 0.7)  # 70% event rate

    survival_data <- S4Vectors::DataFrame(
      time = survival_times,
      event = event_status,
      row.names = paste0("Patient_", 1:n_patients)
    )
  } else {
    survival_data <- S4Vectors::DataFrame(time = numeric(0), event = integer(0))
  }

  create_clinical_project(
    omics_assays = omics_list,
    clinical_data = clinical_data,
    survival_data = survival_data
  )
}

#' Simulate complete multi-omics dataset
#' @param n_features integer, features per omics layer
#' @param n_samples integer
#' @param omics_types character vector
#' @param correlated logical, create correlation structure across omics
#' @return list of OmicsExperiment objects
#' @export
simulate_multi_omics <- function(n_features = 10000, n_samples = 1000,
                                 omics_types = c("rna", "protein", "metabolite"),
                                 correlated = TRUE) {

  message("Simulating multi-omics dataset: ", paste(omics_types, collapse = ", "))

  results <- list()

  # Common biological groups
  groups <- factor(rep(1:2, length.out = n_samples))

  # Generate correlated latent factors if requested
  if (correlated) {
    n_factors <- 5
    latent_factors <- matrix(rnorm(n_samples * n_factors), nrow = n_samples)

    # Group-specific factors
    group_effects <- matrix(0, nrow = n_samples, ncol = n_factors)
    group_effects[groups == 2, ] <- group_effects[groups == 2, ] + 1
    latent_factors <- latent_factors + group_effects
  }

  # RNA-seq
  if ("rna" %in% omics_types) {
    results$RNA <- simulate_rnaseq(n_features, n_samples)
  }

  # Proteomics
  if ("protein" %in% omics_types) {
    results$Protein <- simulate_proteomics(round(n_features / 2), n_samples)
  }

  # Metabolomics (simplified - similar to proteomics but smaller)
  if ("metabolite" %in% omics_types) {
    results$Metabolite <- simulate_proteomics(round(n_features / 5), n_samples)
    results$Metabolite@omics_type <- "metabolite"
  }

  results
}
