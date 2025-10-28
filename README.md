# OmniOmicsR: An Integrated R Framework for Reproducible Multi-Omics Data Analysis and Visualization

[![R](https://img.shields.io/badge/R-4.3%2B-blue.svg)](https://www.r-project.org/)
[![License: GPL-3](https://img.shields.io/badge/license-GPL--3-brightgreen.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[![Bioconductor](https://img.shields.io/badge/Bioconductor-Compatible-orange.svg)](https://www.bioconductor.org/)
[![Status](https://img.shields.io/badge/build-passing-success.svg)]()

---

## 🧬 Project Overview

**OmniOmicsR** delivers an end-to-end, reproducible **multi-omics analysis workflow** in R, seamlessly integrating RNA sequencing, proteomics, and metabolomics datasets within a unified S4 object system.  

Building on Bioconductor’s core infrastructure, OmniOmicsR introduces two key abstractions:

- **`OmicsExperiment`** — encapsulates assay matrices, feature/sample metadata, and processing logs.  
- **`OmniProject`** — provides a project-level container linking multiple `OmicsExperiment` instances, reference designs, and integration results.  

From raw data ingestion to **quality control**, **normalization**, **batch correction**, **multi-omics integration** (DIABLO / MOFA2 / RGCCA), and **statistical testing**, OmniOmicsR ensures that teams can perform rigorous, reproducible, and extensible analyses in one coherent framework.

---
## Author

**Dustin**--**Turning data into decisions, algorithms into value.**

---

## ⚙️ Installation

Follow these steps on **Ubuntu / WSL2** to prepare the runtime and load the development version.

### 1️⃣ System Preparation

```bash
sudo apt update && sudo apt install -y \
  r-base \
  libcurl4-openssl-dev \
  libxml2-dev \
  libssl-dev \
  libharfbuzz-dev \
  libfribidi-dev
````

### 2️⃣ R Dependencies

Launch **R** and install the required dependencies:

```r
install.packages(c("data.table","Matrix","Rcpp","ggplot2","pkgload","devtools"))
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install(c("SummarizedExperiment","MultiAssayExperiment",
                       "BiocParallel","DelayedArray","S4Vectors"))
devtools::load_all(".")
```

> 💡 Optional: install additional toolkits (`edgeR`, `DESeq2`, `limma`, `mixOmics`, `MOFA2`, `RGCCA`) to unlock full functionality for differential testing and cross-omics integration.

---

## 🚀 Quick Start Example

Below is a minimal example demonstrating how to load a toy RNA expression dataset, run QC, normalize counts, and visualize results.

```r
library(OmniOmicsR)

# Example data bundled with the package
f <- system.file("extdata/example_counts.csv", package = "OmniOmicsR")

rna <- read_omics_matrix(f, omics_type = "rna") |>
  qc_basic() |>
  normalize_tmm()

plot_qc(rna)
```

If successful, this pipeline prints:

```
OmicsExperiment<rna> with 4 features and 4 samples
```

and generates a **QC plot** named `qc_plot.png` in your working directory.

> 📂 Explore `inst/extdata/` for additional toy datasets and templates.

---

## 🧩 Package Structure

| Directory                                                    | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **R/classes-OmicsExperiment.R**                              | Defines `OmicsExperiment` and `OmniProject` S4 classes.      |
| **R/io-readers.R**                                           | Data import for RNA, proteomics (MaxQuant), metabolomics (mzTab), and Seurat fallbacks. |
| **R/qc-metrics.R**, **R/qc-scrna.R**                         | Core quality control metrics and scRNA-seq support.          |
| **R/preprocess-normalize.R**, **R/preprocess-batch.R**       | Normalization (TMM, quantile) and batch correction (ComBat, MNN). |
| **R/integrate-multiomics.R**                                 | Integration backends for DIABLO, MOFA2, and RGCCA.           |
| **R/stats-diffexp.R**, **R/stats-association.R**, **R/stats-enrich.R** | Differential expression, association testing, and pathway enrichment. |
| **R/viz-qc.R**, **R/viz-integrations.R**                     | Visualization utilities for QC and cross-omics embedding.    |
| **src/utils.cpp**                                            | C++ accelerated matrix operations.                           |
| **inst/scripts/default.qmd**                                 | Quarto reporting template.                                   |
| **inst/extdata/**                                            | Example datasets for demo workflows.                         |

---

## 🧪 Development and Testing

Run the built-in test suite to verify your installation:

```r
devtools::test()
```

Regenerate documentation:

```r
devtools::document()
```

Rebuild and reload the package:

```r
devtools::build()
devtools::load_all(".")
```

---

## 📊 Outputs

* **QC metrics** and **plots** (e.g., `qc_plot.png`)
* **Normalized expression matrices** via `normalize_tmm()`
* **Integrated latent representations** for DIABLO/MOFA2/RGCCA
* **Differential analysis tables** and **enrichment results**
* **Project-level reports** generated from Quarto templates

---

## Acknowledgments

This work was made possible through the open-source Bioconductor ecosystem and the invaluable inspiration from the research community.

Most importantly, I want to thank the most important person in my life—**Yanyan**—for always being by my side.
She supported me when I faced difficulties; encouraged me when I was lost; and believed in me even when I doubted myself.
Every bit of progress I've made stems from her unwavering belief in my ability.This project may be the product of my hard work, but it's also the product of her strength—and for that, I am deeply grateful.

---

© 2025 Dustin. Released under the GPL-3 License.