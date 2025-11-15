# OmniOmicsR:A Unified, Scalable, and Generative Framework for Next-Generation Multi-Omics and Clinical Integration
<img src="images/OmniOmicsR.png" alt="OmniOmicsR" width="400" height="200">
[![R Version](https://img.shields.io/badge/R-4.3%2B-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0.0-success.svg)](DESCRIPTION)

---

## Overview

**OmniOmicsR v2.0** is a comprehensive R package for end-to-end multi-omics data analysis, integrating machine learning, Bayesian inference, spatial omics, single-cell analysis, and clinical outcomes within a unified, reproducible framework.

Built on Bioconductor's S4 class system, OmniOmicsR enables:

- 🧬 **Multi-omics integration** (RNA-seq, proteomics, metabolomics, spatial, single-cell)
- 🤖 **Advanced ML** (VAE, Random Forest, XGBoost, ensemble methods)
- 🧠 **Deep learning** (Graph neural networks, diffusion models via **OmniGraphDiff**)
- 📊 **Sophisticated statistics** (Bayesian inference, network analysis, differential expression)
- 🏥 **Clinical analysis** (survival, biomarkers, patient stratification)

---

## Key Features

### Multi-Omics Integration
- Native support for RNA-seq, proteomics, metabolomics, and Seurat objects
- Integration methods: DIABLO, MOFA2, RGCCA, canonical correlation
- Batch correction: ComBat, MNN, Harmony

### Machine Learning & Deep Learning
- **Variational Autoencoders (VAE)** for dimensionality reduction
- **Ensemble methods**: Random Forest, XGBoost, stacking
- **Feature selection**: LASSO, Elastic Net, Boruta, mRMR
- **OmniGraphDiff**: Hierarchical graph neural networks, graph-VAE, diffusion models
  - See [omnigraphdiff/README.md](omnigraphdiff/README.md) for details

### Statistical Analysis
- Bayesian inference (Stan/JAGS backends)
- Network analysis (WGCNA, GENIE3, PPI enrichment)
- Differential expression (edgeR, DESeq2, limma)

### Spatial & Single-Cell Analysis
- Spatial transcriptomics with coordinate-based analyses
- Single-cell multi-omics (CITE-seq, scATAC-seq, multiome)
- Trajectory inference and cell-cell communication

### Clinical Integration
- Survival analysis (Cox models, Kaplan-Meier)
- Biomarker discovery and validation
- Patient stratification and outcome prediction

---

## Installation

### Quick Install

```r
# Install dependencies
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install(c("SummarizedExperiment", "MultiAssayExperiment"))

# Install OmniOmicsR
devtools::install_github("your-username/OmniOmicsR", dependencies = TRUE)
```

### System Requirements
- **R:** ≥ 4.3.0
- **OS:** Linux, macOS, or Windows (WSL2 recommended)
- **Memory:** 8GB minimum, 16GB+ recommended

For detailed installation instructions and optional packages, see [full documentation](#documentation).

---

## Quick Start

### Basic Workflow

```r
library(OmniOmicsR)

# Load and process data
rna_data <- read_omics_matrix("counts.csv", omics_type = "rna") |>
  qc_basic() |>
  normalize_tmm() |>
  normalize_vst()

# Visualization
plot_pca(rna_data, color_by = "group")

# Differential expression
dea_results <- dea_deseq2(rna_data, design = ~group)
sig_genes <- dea_results[dea_results$padj < 0.05, ]
```

### Multi-Omics Integration

```r
# Create project
omics_list <- list(RNA = rna_data, Protein = protein_data)
project <- as_op(omics_list)

# DIABLO integration
integration <- integrate_diablo(omics_list, outcome = outcome, ncomp = 3)
plot_circos_integrate(integration)
```

### Clinical Analysis

```r
# Create clinical project
clinical_project <- create_clinical_project(
  omics_assays = omics_list,
  clinical_data = clinical_df,
  survival_data = survival_df
)

# Survival analysis and biomarker discovery
survival_result <- clinical_survival(clinical_project)
biomarkers <- clinical_biomarkers(clinical_project, outcome = stage)
```

### Deep Learning with OmniGraphDiff

```r
# Use from R via reticulate
library(reticulate)
use_virtualenv("~/omnigraphdiff_env")

ogd <- import("omnigraphdiff")
model <- ogd$train$train_model(config_path = "config.yaml")

# Extract embeddings for downstream analysis
embeddings <- model$encode(list(rna = rna_mat, protein = protein_mat))
```

See [omnigraphdiff/README.md](omnigraphdiff/README.md) for complete OmniGraphDiff documentation.

---

## Documentation

### Core Documentation
- **[Getting Started](vignettes/getting_started.qmd)** — Basic workflow
- **[Architecture](ARCHITECTURE.md)** — Design principles
- **[Multi-Omics Tutorial](vignettes/multiomics_integration.qmd)** — Integration methods

### OmniGraphDiff (Deep Learning Module)
- **[Overview & Installation](omnigraphdiff/README.md)** — Setup and training guide
- **[Model Design](omnigraphdiff/MODEL_DESIGN.md)** — Mathematical formulation
- **[Architecture](omnigraphdiff/ARCHITECTURE.md)** — System design

### Function Reference
```r
?OmniOmicsR          # Package overview
?OmicsExperiment     # Core S4 class
?train_vae           # VAE training
?clinical_survival   # Survival analysis
```

---

## Package Structure

```
OmniOmicsR/
├── R/                    # R source code
├── src/                  # C++ source (Rcpp)
├── inst/                 # Example data & templates
├── tests/                # Test suite
├── vignettes/            # Tutorials
├── omnigraphdiff/        # Deep learning module (PyTorch)
│   ├── omnigraphdiff/    # Python package
│   ├── cpp_backend/      # C++ graph operations
│   ├── examples/         # Training examples
│   └── *.md              # Documentation
└── DESCRIPTION           # Package metadata
```

---

## Testing & Performance

```r
# Run tests
devtools::test()

# Quick validation (~5 min)
source(system.file("scripts/quick_validation.R", package = "OmniOmicsR"))

# Benchmark
benchmark_results <- benchmark_all(n_features = 10000, n_samples = 1000)
```

**Performance** (10K features × 1K samples):
- TMM normalization: <1s
- VAE training (10 epochs): ~23s
- Random Forest: ~15s
- Multi-omics integration: ~40s

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```r
# Development workflow
git clone https://github.com/dxpython/OmniOmicsR.git
git checkout -b feature/your-feature
devtools::test()
devtools::check()
```

---

## Citation

If you use OmniOmicsR in your research, please cite:

```
[Citation information to be added]
```

---

## Acknowledgments

OmniOmicsR builds on excellent work from the Bioconductor community:

**Core packages:** SummarizedExperiment, MultiAssayExperiment, edgeR, DESeq2, limma, mixOmics, MOFA2, WGCNA, Seurat

**ML/DL frameworks:** keras/TensorFlow, PyTorch, ranger, xgboost

**Personal thanks:** To Yanyan, whose unwavering support made this project possible.

---

## License

MIT License © 2025 Dustin Dong

---

<div align="center">

**Built with ❤️ for the multi-omics research community**

</div>
