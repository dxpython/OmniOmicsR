# OmniOmicsR:A Unified, Scalable, and Generative Framework for Next-Generation Multi-Omics and Clinical Integration

![OmniOmicsR](images/OmniOmicsR.png)


<div align="center">

[![R Version](https://img.shields.io/badge/R-4.3%2B-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Bioconductor](https://img.shields.io/badge/Bioconductor-Compatible-orange.svg)](https://www.bioconductor.org/)
[![Version](https://img.shields.io/badge/Version-2.0.0-success.svg)](DESCRIPTION)
[![Build Status](https://img.shields.io/badge/Build-Passing-success.svg)]()

**[Documentation](#-documentation)** • **[Installation](#%EF%B8%8F-installation)** • **[Quick Start](#-quick-start)** • **[Features](#-key-features)** • **[Citation](#-citation)**

---

## 🔬 Overview

**OmniOmicsR v2.0** is a comprehensive, production-grade R package for **end-to-end multi-omics data analysis**, seamlessly integrating cutting-edge machine learning, Bayesian inference, spatial omics, single-cell multi-omics, and clinical outcome analysis—all within a unified, reproducible framework.

Built on Bioconductor's robust S4 class system, OmniOmicsR provides researchers and bioinformaticians with powerful tools to:

- 🧬 **Integrate multiple omics modalities** (RNA-seq, proteomics, metabolomics, spatial, single-cell)
- 🤖 **Apply advanced machine learning** (VAE, Random Forest, XGBoost, ensemble methods)
- 🧠 **Leverage deep learning** (Hierarchical graph neural networks, graph-VAE, diffusion models via **OmniGraphDiff**)
- 📊 **Perform sophisticated statistics** (Bayesian inference, network analysis, differential expression)
- 🏥 **Connect omics to clinical outcomes** (survival analysis, biomarker discovery, patient stratification)
- 🎨 **Generate publication-ready visualizations** and reproducible reports

### Core Design Principles

**OmniOmicsR** is built around two fundamental S4 class abstractions:

- **`OmicsExperiment`** — Encapsulates assay data, feature/sample metadata, and processing history for a single omics modality
- **`OmniProject`** — Project-level container orchestrating multiple experiments, experimental designs, and integration results
- **Extended classes** — `SpatialOmicsExperiment`, `SingleCellMultiOmicsExperiment`, `ClinicalOmicsProject` for specialized analyses

From raw data import through quality control, normalization, batch correction, multi-omics integration, statistical testing, and clinical modeling—**every step is reproducible, extensible, and production-ready**.

---

## ✨ Key Features

### 🧬 **Multi-Omics Data Integration**

- **Native support** for RNA-seq, proteomics (MaxQuant), metabolomics (mzTab), and Seurat objects
- **Multi-modal integration** via DIABLO, MOFA2, RGCCA, and canonical correlation analysis
- **Cross-omics correlation** and visualization with circos plots
- **Batch correction** using ComBat, MNN, and Harmony

### 🤖 **Advanced Machine Learning**

- **Variational Autoencoders (VAE)** for deep dimensionality reduction and integration
- **Ensemble methods** including Random Forest, XGBoost, and stacking
- **Advanced feature selection** with LASSO, Elastic Net, Boruta, stability selection, and mRMR
- **Automated hyperparameter tuning** and cross-validation

### 📊 **Sophisticated Statistical Analysis**

- **Bayesian inference** for differential expression and network learning (Stan/JAGS backends)
- **Network analysis** including WGCNA, gene regulatory networks (GENIE3), and PPI enrichment
- **Meta-analysis** for cross-study integration
- **Association testing** with linear mixed models and correlation analysis

### 🗺️ **Spatial Omics Analysis**

- **Spatial transcriptomics** support with coordinate-based analyses
- **Spatial variable feature detection** using Moran's I and Geary's C statistics
- **Spatial clustering** and domain identification
- **Spatial trajectory inference** and pseudotime analysis
- **Spatial cell-cell communication** prediction

### 🔬 **Single-Cell Multi-Omics**

- **CITE-seq, scATAC-seq, and multiome** data support
- **Weighted nearest neighbor (WNN)** integration across modalities
- **Trajectory inference** (Slingshot-like algorithms)
- **Cell-cell communication** prediction with ligand-receptor databases

### 🏥 **Clinical Integration**

- **Survival analysis** with Cox proportional hazards models
- **Biomarker discovery** pipelines with multiple selection methods
- **Patient stratification** and clustering
- **Clinical outcome prediction** with ML models
- **Kaplan-Meier curves** and time-to-event analysis

### 🧠 **Deep Learning: OmniGraphDiff**

- **Hierarchical graph neural networks** for multi-scale omics integration
- **Graph-VAE** with modality-specific and shared latent representations
- **Graph-conditioned diffusion models** (DDPM) for data generation and imputation
- **Multi-GPU training** with mixed precision and distributed data parallelism
- **C++ backend** with optimized sparse graph operations (CSR/CSC, SpMM, Laplacian)
- **Clinical prediction** from learned embeddings (survival, classification, stratification)
- **Seamless R integration** via reticulate for use within OmniOmicsR workflows

👉 **See [omnigraphdiff/README.md](omnigraphdiff/README.md) for complete documentation, installation, and training guides.**

### 🛠️ **Production Features**

- **Comprehensive simulation engine** for realistic multi-omics data generation
- **Benchmarking suite** for performance testing and validation
- **Processing logs** for complete reproducibility
- **Automated reporting** with Quarto/RMarkdown templates
- **Parallel processing** support via BiocParallel
- **Memory-efficient** implementations with optional sparse matrices

---

## ⚙️ Installation

### System Requirements

- **R version:** ≥ 4.3.0
- **Operating System:** Linux, macOS, or Windows (WSL2 recommended)
- **Memory:** Minimum 8GB RAM (16GB+ recommended for large datasets)

### 1️⃣ System Dependencies (Ubuntu/Debian)

```bash
sudo apt update && sudo apt install -y \
  r-base \
  libcurl4-openssl-dev \
  libxml2-dev \
  libssl-dev \
  libharfbuzz-dev \
  libfribidi-dev \
  libgit2-dev \
  libfontconfig1-dev
```

### 2️⃣ Install Core R Dependencies

```r
# Install from CRAN
install.packages(c(
  "data.table", "Matrix", "Rcpp", "ggplot2",
  "devtools", "testthat", "roxygen2"
))

# Install Bioconductor packages
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install(c(
  "SummarizedExperiment",
  "MultiAssayExperiment",
  "BiocParallel",
  "DelayedArray",
  "S4Vectors"
))
```

### 3️⃣ Install OmniOmicsR

**From GitHub (Latest Development Version):**

```r
devtools::install_github("your-username/OmniOmicsR", dependencies = TRUE)
```

**Or clone and install locally:**

```bash
git clone https://github.com/your-username/OmniOmicsR.git
cd OmniOmicsR
```

```r
devtools::install(".", dependencies = TRUE)
```

### 4️⃣ Optional Enhancement Packages

Install optional packages to unlock full functionality:

```r
# Differential expression and normalization
BiocManager::install(c("edgeR", "DESeq2", "limma", "sva"))

# Multi-omics integration
BiocManager::install(c("mixOmics", "MOFA2", "RGCCA"))

# Machine learning
install.packages(c("ranger", "xgboost", "glmnet", "Boruta"))

# Bayesian inference
install.packages(c("rstan", "bnlearn"))

# Network analysis
BiocManager::install("WGCNA")
install.packages(c("igraph", "GENIE3"))

# Spatial and single-cell
BiocManager::install(c("Seurat", "spatstat", "SingleCellExperiment"))

# Clinical analysis
install.packages(c("survival", "survminer"))

# Deep learning (optional)
install.packages(c("keras", "tensorflow", "reticulate"))
```

### 5️⃣ Verify Installation

```r
library(OmniOmicsR)
packageVersion("OmniOmicsR")  # Should show 2.0.0

# Run quick validation test
source(system.file("scripts/quick_validation.R", package = "OmniOmicsR"))
```

### 6️⃣ (Optional) Install OmniGraphDiff for Deep Learning

If you want to use the advanced deep learning features (hierarchical graph neural networks, diffusion models):

```bash
# Navigate to OmniGraphDiff directory
cd omnigraphdiff

# Follow the installation guide
# See omnigraphdiff/README.md for complete instructions
```

**Quick setup:**
1. Create Python environment (Python 3.8+)
2. Install PyTorch with CUDA support (if available)
3. Build C++ backend with pybind11
4. Install OmniGraphDiff Python package

**Full instructions:** See [omnigraphdiff/README.md](omnigraphdiff/README.md#build--install)

---

## 🚀 Quick Start

### Basic Multi-Omics Workflow

```r
library(OmniOmicsR)

# ═══════════════════════════════════════════════════════════
# 1. Load Example RNA-seq Data
# ═══════════════════════════════════════════════════════════
counts_file <- system.file("extdata/example_counts.csv", package = "OmniOmicsR")
rna_data <- read_omics_matrix(counts_file, omics_type = "rna")

# ═══════════════════════════════════════════════════════════
# 2. Quality Control and Normalization
# ═══════════════════════════════════════════════════════════
rna_data <- rna_data |>
  qc_basic() |>
  normalize_tmm() |>
  normalize_vst()

# ═══════════════════════════════════════════════════════════
# 3. Visualization
# ═══════════════════════════════════════════════════════════
plot_qc(rna_data)
plot_pca(rna_data, color_by = "group")

# ═══════════════════════════════════════════════════════════
# 4. Differential Expression Analysis
# ═══════════════════════════════════════════════════════════
dea_results <- dea_deseq2(rna_data, design = ~group)

# Filter significant genes
sig_genes <- dea_results[dea_results$padj < 0.05, ]
head(sig_genes)
```

### Advanced Example: Multi-Omics Integration

```r
# ═══════════════════════════════════════════════════════════
# 1. Create Multi-Omics Project
# ═══════════════════════════════════════════════════════════
rna <- read_omics_matrix("rna_counts.csv", omics_type = "rna")
protein <- read_maxquant("proteinGroups.txt")

omics_list <- list(RNA = rna, Protein = protein)
project <- as_op(omics_list)

# ═══════════════════════════════════════════════════════════
# 2. Multi-Omics Integration with DIABLO
# ═══════════════════════════════════════════════════════════
outcome <- rep(c("Control", "Treatment"), each = 50)

integration_result <- integrate_diablo(
  omics_list,
  outcome = outcome,
  ncomp = 3
)

# ═══════════════════════════════════════════════════════════
# 3. Visualize Integrated Results
# ═══════════════════════════════════════════════════════════
plot_circos_integrate(integration_result)
```

### Machine Learning Example: VAE Integration

```r
# ═══════════════════════════════════════════════════════════
# Train Variational Autoencoder for Dimensionality Reduction
# ═══════════════════════════════════════════════════════════
vae_result <- train_vae(
  rna_data,
  latent_dim = 32,
  hidden_dims = c(512, 256, 128),
  epochs = 100
)

# Access latent representation
latent_space <- vae_result$latent
head(latent_space)

# Multi-omics VAE integration
multi_vae <- integrate_vae_multiomics(
  omics_list,
  latent_dim = 32,
  shared_dim = 16
)
```

### Clinical Integration Example

```r
# ═══════════════════════════════════════════════════════════
# Create Clinical Omics Project
# ═══════════════════════════════════════════════════════════
clinical_data <- data.frame(
  patient_id = paste0("P", 1:100),
  age = rnorm(100, 60, 10),
  sex = sample(c("M", "F"), 100, replace = TRUE),
  stage = sample(1:4, 100, replace = TRUE)
)

survival_data <- data.frame(
  time = rexp(100, rate = 0.01),
  event = rbinom(100, 1, 0.7)
)

clinical_project <- create_clinical_project(
  omics_assays = omics_list,
  clinical_data = clinical_data,
  survival_data = survival_data
)

# ═══════════════════════════════════════════════════════════
# Survival Analysis
# ═══════════════════════════════════════════════════════════
survival_result <- clinical_survival(
  clinical_project,
  omics_assay = "RNA"
)

# ═══════════════════════════════════════════════════════════
# Biomarker Discovery
# ═══════════════════════════════════════════════════════════
biomarkers <- clinical_biomarkers(
  clinical_project,
  outcome = clinical_data$stage,
  method = "elastic_net"
)

# ═══════════════════════════════════════════════════════════
# Patient Stratification
# ═══════════════════════════════════════════════════════════
stratification <- clinical_stratify(
  clinical_project,
  n_groups = 3,
  method = "kmeans"
)

plot_survival(clinical_project, groups = stratification$stratification)
```

### Deep Learning Example: OmniGraphDiff from R

**OmniGraphDiff** is a PyTorch-based deep learning framework for multi-omics integration using hierarchical graph neural networks, variational autoencoders, and diffusion models. It can be seamlessly used from R:

```r
# ═══════════════════════════════════════════════════════════
# 1. Setup Python Environment (One-time)
# ═══════════════════════════════════════════════════════════
library(reticulate)

# Point to OmniGraphDiff Python environment
use_virtualenv("~/omnigraphdiff_env")
# Or use conda: use_condaenv("omnigraphdiff")

# Import OmniGraphDiff
ogd <- import("omnigraphdiff")

# ═══════════════════════════════════════════════════════════
# 2. Prepare Multi-Omics Data
# ═══════════════════════════════════════════════════════════
# Assuming you have processed data in OmniOmicsR
rna_mat <- assay(rna_data, "normalized")  # genes × samples
protein_mat <- assay(protein_data, "normalized")

# Save to HDF5 format for OmniGraphDiff
library(rhdf5)
h5createFile("multiomics_data.h5")
h5write(t(rna_mat), "multiomics_data.h5", "rna")  # samples × genes
h5write(t(protein_mat), "multiomics_data.h5", "protein")
h5write(rownames(rna_mat), "multiomics_data.h5", "rna_features")
h5write(rownames(protein_mat), "multiomics_data.h5", "protein_features")

# ═══════════════════════════════════════════════════════════
# 3. Train OmniGraphDiff Model
# ═══════════════════════════════════════════════════════════
# Create configuration
config <- list(
  model = list(
    hidden_dim = 256L,
    latent_dim = 64L,
    num_layers = 3L
  ),
  training = list(
    batch_size = 32L,
    learning_rate = 0.001,
    num_epochs = 100L
  ),
  data = list(
    path = "multiomics_data.h5",
    modalities = list("rna", "protein")
  )
)

# Save config as YAML
yaml::write_yaml(config, "config.yaml")

# Train model (can also use system() to run Python script)
ogd$train$train_model(config_path = "config.yaml", output_dir = "results")

# ═══════════════════════════════════════════════════════════
# 4. Extract Latent Embeddings
# ═══════════════════════════════════════════════════════════
# Load trained model
model <- ogd$models$load_model("results/checkpoints/best_model.pt")

# Extract latent representations
embeddings <- model$encode(list(rna = rna_mat, protein = protein_mat))

# Convert to R data frame
latent_df <- as.data.frame(py_to_r(embeddings$shared))
rownames(latent_df) <- colnames(rna_mat)

# ═══════════════════════════════════════════════════════════
# 5. Use Embeddings in OmniOmicsR
# ═══════════════════════════════════════════════════════════
# Add embeddings back to OmniOmicsR project
project@integration_results[["omnigraphdiff"]] <- list(
  latent_shared = latent_df,
  latent_rna = as.data.frame(py_to_r(embeddings$rna_specific)),
  latent_protein = as.data.frame(py_to_r(embeddings$protein_specific))
)

# Use for downstream analysis
plot_pca(latent_df, color_by = clinical_data$stage)
survival_from_embeddings <- clinical_survival(project, embeddings = latent_df)
```

**For complete OmniGraphDiff documentation:**
- **Installation guide:** See [omnigraphdiff/README.md](omnigraphdiff/README.md#build--install)
- **Training on your data:** See [omnigraphdiff/README.md](omnigraphdiff/README.md#how-to-train-omnigraphdiff-on-your-own-multi-omics-data)
- **Model architecture:** See [omnigraphdiff/MODEL_DESIGN.md](omnigraphdiff/MODEL_DESIGN.md)
- **System architecture:** See [omnigraphdiff/ARCHITECTURE.md](omnigraphdiff/ARCHITECTURE.md)

---

## 📚 Documentation

### Getting Started

- **[Getting Started Vignette](vignettes/getting_started.qmd)** — Basic workflow and concepts
- **[Architecture Overview](ARCHITECTURE.md)** — Design principles and class hierarchy
- **[V2.0 Enhancements](V2_ENHANCEMENTS.md)** — What's new in version 2.0

### Advanced Tutorials

- **[RNA-Proteomics Workflow](vignettes/rna_proteomics_workflow.qmd)** — Multi-omics integration
- **[Single-Cell Analysis](vignettes/singlecell_workflow.qmd)** — scRNA-seq and multiome
- **[Multi-Omics Integration](vignettes/multiomics_integration.qmd)** — DIABLO, MOFA2, RGCCA

### Deep Learning Module: OmniGraphDiff

- **[OmniGraphDiff Overview](omnigraphdiff/README.md)** — Complete guide to the PyTorch deep learning module
- **[Model Design](omnigraphdiff/MODEL_DESIGN.md)** — Mathematical formulation of hierarchical graph-VAE and diffusion models
- **[System Architecture](omnigraphdiff/ARCHITECTURE.md)** — Technical architecture and component design
- **[Build & Install](omnigraphdiff/README.md#build--install)** — Setup Python environment and build C++ backend
- **[Training Guide](omnigraphdiff/README.md#how-to-train-omnigraphdiff-on-your-own-multi-omics-data)** — Train on your own data

### Code Quality & Development

- **[Code Review & Cleanup](CODE_REVIEW_AND_CLEANUP.md)** — Code quality analysis
- **[Refactoring Examples](REFACTORING_EXAMPLES.md)** — Best practices and optimizations
- **[Cleanup Action Plan](CLEANUP_ACTION_PLAN.md)** — Development roadmap

### Function Reference

Generate complete function documentation:

```r
?OmniOmicsR  # Package overview
?OmicsExperiment  # Core S4 class
?train_vae  # Variational autoencoder
?ensemble_rf  # Random forest
?spatial_clustering  # Spatial analysis
?clinical_survival  # Survival analysis
```

---

## 🧪 Testing & Validation

### Run Test Suite

```r
# Run all tests
devtools::test()

# Run specific test files
devtools::test(filter = "ml")
devtools::test(filter = "spatial")
```

### Benchmark Performance

```r
# Quick validation (5 min)
source(system.file("scripts/quick_validation.R", package = "OmniOmicsR"))

# Comprehensive benchmark (15-20 min)
source(system.file("scripts/demo_simulation_benchmark.R", package = "OmniOmicsR"))

# Custom benchmark
library(OmniOmicsR)
benchmark_results <- benchmark_all(
  n_features = 10000,
  n_samples = 1000,
  verbose = TRUE
)
```

### Performance Characteristics

**Validated on 10,000 features × 1,000 samples:**

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| RNA-seq simulation | ~18s | ~500MB | Realistic count data |
| TMM normalization | <1s | ~800MB | edgeR backend |
| VAE training (10 epochs) | ~23s | ~1.2GB | PCA fallback: ~20s |
| Random Forest (100 trees) | ~15s | ~1.5GB | ranger backend |
| WGCNA (subset) | ~30s | ~2GB | 500 features |
| Multi-omics integration | ~40s | ~2.5GB | DIABLO |

---

## 📦 Package Structure

```
OmniOmicsR/
├── R/                          # R source code
│   ├── 01-classes.R            # S4 class definitions
│   ├── utils-*.R               # Utility functions
│   ├── io-*.R                  # Data I/O
│   ├── qc-*.R                  # Quality control
│   ├── preprocess-*.R          # Normalization & batch correction
│   ├── stats-*.R               # Statistical methods
│   ├── ml-*.R                  # Machine learning
│   ├── spatial-*.R             # Spatial omics
│   ├── sc-*.R                  # Single-cell multi-omics
│   ├── clinical-*.R            # Clinical integration
│   ├── viz-*.R                 # Visualization
│   ├── simulation-*.R          # Data simulation
│   └── benchmark-*.R           # Benchmarking
├── src/                        # C++ source (Rcpp)
│   └── utils.cpp               # Fast matrix operations
├── inst/                       # Installed files
│   ├── extdata/                # Example datasets
│   ├── scripts/                # Demo scripts
│   └── templates/              # Report templates
├── tests/                      # Test suite
│   └── testthat/
├── vignettes/                  # Tutorials
├── man/                        # Documentation
├── omnigraphdiff/              # Deep learning module (PyTorch)
│   ├── omnigraphdiff/          # Python package
│   │   ├── models/             # GNN, VAE, Diffusion models
│   │   ├── training/           # Trainer, callbacks, DDP
│   │   ├── data/               # DataLoaders, HDF5/NPZ
│   │   ├── losses/             # Multi-objective loss functions
│   │   ├── graphs/             # Graph construction & ops
│   │   └── utils/              # Config, logging, metrics
│   ├── cpp_backend/            # C++ sparse graph operations
│   ├── R/                      # R interface via reticulate
│   ├── examples/               # Training demos & configs
│   ├── tests/                  # Python unit tests
│   ├── MODEL_DESIGN.md         # Mathematical formulation
│   ├── ARCHITECTURE.md         # System design
│   └── README.md               # Complete documentation
└── DESCRIPTION                 # Package metadata
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

```r
# 1. Clone repository
git clone https://github.com/dxpython/OmniOmicsR.git

# 2. Create feature branch
git checkout -b feature/your-feature

# 3. Make changes and test
devtools::test()
devtools::check()

# 4. Document changes
devtools::document()


## 📊 Outputs & Reporting

OmniOmicsR generates comprehensive outputs including:

- ✅ **QC reports** with interactive visualizations
- ✅ **Normalized expression matrices** (multiple methods)
- ✅ **Integrated latent representations** (VAE, DIABLO, MOFA2)
- ✅ **Differential analysis tables** with FDR correction
- ✅ **Pathway enrichment results** (GO, KEGG, Reactome)
- ✅ **Clinical prediction models** with cross-validation
- ✅ **Publication-ready figures** (ggplot2-based)
- ✅ **Automated HTML reports** (Quarto/RMarkdown)

### Export Results

```r
# Export project with all results
save_project(project, "my_analysis.rds")

# Generate HTML report
export_op_report(project, output_file = "report.html")

# Replay processing steps
loaded_project <- load_project("my_analysis.rds")
replay(loaded_project)  # Re-run entire pipeline
```

---

##  Acknowledgments

**OmniOmicsR** builds upon the remarkable work of the open-source Bioconductor community. We are deeply grateful for the foundational packages and methodologies that made this work possible.

### Technical Foundations

This package integrates and extends numerous excellent tools:

- **Bioconductor Core:** SummarizedExperiment, MultiAssayExperiment, S4Vectors
- **Differential Expression:** edgeR, DESeq2, limma
- **Multi-Omics Integration:** mixOmics (DIABLO), MOFA2, RGCCA
- **Machine Learning:** keras/TensorFlow, ranger, xgboost, glmnet
- **Deep Learning (OmniGraphDiff):** PyTorch, PyTorch Geometric, pybind11, lifelines
- **Network Analysis:** WGCNA, igraph, GENIE3
- **Spatial Analysis:** Seurat, spatstat, Giotto
- **Statistics:** survival, rstan, metafor

### Personal Acknowledgment

Most importantly, I want to thank **Yanyan**—the most important person in my life. Her unwavering support, encouragement, and belief in my abilities have been the foundation of every achievement. This project exists because of her strength and dedication. For that, I am deeply grateful.

---

## 📜 License

OmniOmicsR is released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2025 Dustin Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
```

---

<div align="center">
**Built with ❤️ for the multi-omics research community**

*Turning data into decisions, algorithms into value*

**© 2025 Dustin Dong • MIT License**

</div>
