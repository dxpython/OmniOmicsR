# OmniGraphDiff - Complete Project Structure

```
omnigraphdiff/
├── README.md                           # Main project documentation
├── ARCHITECTURE.md                     # Detailed architecture documentation
├── MODEL_DESIGN.md                     # Mathematical model formulation
├── TRAINING_GUIDE.md                   # Training instructions
├── API_REFERENCE.md                    # API documentation
├── LICENSE                             # MIT or Apache 2.0
├── setup.py                            # Python package setup
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment
│
├── cpp_backend/                        # C++ high-performance backend
│   ├── CMakeLists.txt                  # CMake build configuration
│   ├── include/
│   │   ├── omnigraph/
│   │   │   ├── sparse_graph.hpp        # CSR/CSC sparse graph
│   │   │   ├── graph_ops.hpp           # SpMM, Laplacian operations
│   │   │   ├── message_passing.hpp     # Message passing primitives
│   │   │   ├── sampling.hpp            # Neighborhood sampling
│   │   │   └── types.hpp               # Common types and utilities
│   ├── src/
│   │   ├── sparse_graph.cpp
│   │   ├── graph_ops.cpp
│   │   ├── message_passing.cpp
│   │   ├── sampling.cpp
│   │   └── bindings.cpp                # pybind11 Python bindings
│   └── tests/
│       ├── test_sparse_graph.cpp
│       └── test_graph_ops.cpp
│
├── omnigraphdiff/                      # Python package
│   ├── __init__.py
│   ├── version.py
│   │
│   ├── models/                         # Neural network models
│   │   ├── __init__.py
│   │   ├── omnigraph_vae.py           # Hierarchical Graph-VAE
│   │   ├── omnigraph_diffusion.py     # Graph-conditioned diffusion
│   │   ├── encoders.py                # Modality encoders
│   │   ├── decoders.py                # Modality decoders
│   │   ├── gnn_layers.py              # GNN layers (GCN, GAT, GraphSAGE)
│   │   ├── cross_attention.py         # Cross-level attention
│   │   └── clinical_head.py           # Cox/classification head
│   │
│   ├── losses/                         # Loss functions
│   │   ├── __init__.py
│   │   ├── reconstruction.py          # Reconstruction losses
│   │   ├── kl_divergence.py           # KL divergence
│   │   ├── graph_regularization.py    # Laplacian smoothness
│   │   ├── clinical_loss.py           # Cox/classification loss
│   │   ├── contrastive.py             # InfoNCE contrastive loss
│   │   └── composite_loss.py          # Combined multi-objective loss
│   │
│   ├── data/                           # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py                 # Multi-omics dataset class
│   │   ├── graph_builder.py           # Graph construction utilities
│   │   ├── preprocessing.py           # Data preprocessing
│   │   ├── augmentation.py            # Data augmentation
│   │   └── loaders.py                 # DataLoader utilities
│   │
│   ├── training/                       # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Main training loop
│   │   ├── callbacks.py               # Training callbacks
│   │   ├── lr_scheduler.py            # Learning rate schedules
│   │   └── early_stopping.py          # Early stopping logic
│   │
│   ├── evaluation/                     # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── survival_metrics.py        # C-index, log-rank
│   │   ├── clustering_metrics.py      # ARI, NMI, silhouette
│   │   ├── reconstruction_metrics.py  # MSE, correlation
│   │   └── integration_quality.py     # Cross-modal alignment
│   │
│   ├── visualization/                  # Visualization tools
│   │   ├── __init__.py
│   │   ├── latent_space.py           # UMAP, t-SNE plots
│   │   ├── survival_curves.py        # Kaplan-Meier plots
│   │   ├── graph_viz.py              # Network visualization
│   │   └── feature_importance.py     # Attention/importance plots
│   │
│   └── utils/                          # Utility functions
│       ├── __init__.py
│       ├── config.py                  # Configuration management
│       ├── logging.py                 # Logging utilities
│       ├── reproducibility.py         # Seed setting, determinism
│       └── io.py                      # Save/load utilities
│
├── configs/                            # Configuration files
│   ├── default.yaml                   # Default configuration
│   ├── tcga_pancancer.yaml           # TCGA-specific config
│   ├── spatial_transcriptomics.yaml  # Spatial data config
│   └── ablation_studies.yaml         # Ablation experiment configs
│
├── scripts/                            # Executable scripts
│   ├── train_omnigraphdiff.py        # Main training script
│   ├── evaluate.py                    # Evaluation script
│   ├── benchmark_against_baselines.py # Benchmark comparison
│   ├── build_graphs.py               # Graph construction
│   └── export_results.py             # Export to R-compatible format
│
├── benchmarks/                         # Benchmarking code
│   ├── __init__.py
│   ├── run_mofa2.R                   # MOFA2 baseline
│   ├── run_diablo.R                  # DIABLO baseline
│   ├── run_rgcca.R                   # RGCCA baseline
│   ├── run_pca_baseline.py           # PCA + Cox/RF
│   ├── comparison_metrics.py         # Unified metrics
│   └── generate_report.py            # Benchmark report generator
│
├── r_interface/                        # R integration layer
│   ├── DESCRIPTION                    # R package metadata
│   ├── NAMESPACE                      # R exports
│   ├── R/
│   │   ├── omnigraphdiff.R           # Main R interface
│   │   ├── fit_omnigraphdiff.R       # Model fitting wrapper
│   │   ├── predict_omnigraphdiff.R   # Prediction wrapper
│   │   ├── plot_omnigraph.R          # Plotting functions
│   │   └── utils.R                    # R utilities
│   ├── man/                           # R documentation
│   └── tests/
│       └── testthat/
│
├── experiments/                        # Experiment scripts
│   ├── tcga_pancancer/               # TCGA experiments
│   │   ├── preprocess_tcga.py
│   │   ├── train_tcga.sh
│   │   └── analyze_results.py
│   ├── cptac_proteomics/             # CPTAC experiments
│   ├── multiome_10x/                 # 10X multiome
│   └── spatial_transcriptomics/      # Spatial data
│
├── docs/                               # Documentation
│   ├── source/
│   │   ├── conf.py                    # Sphinx config
│   │   ├── index.rst
│   │   ├── installation.rst
│   │   ├── quickstart.rst
│   │   ├── api/
│   │   ├── tutorials/
│   │   └── mathematical_formulation.rst
│   └── build/
│
├── tests/                              # Python tests
│   ├── conftest.py                    # Pytest configuration
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_data.py
│   ├── test_training.py
│   └── test_integration.py
│
├── data/                               # Example/test data
│   ├── example_multiomics.h5         # Small test dataset
│   └── README.md
│
└── outputs/                            # Training outputs (gitignored)
    ├── checkpoints/
    ├── logs/
    ├── results/
    └── figures/
```

## Key Design Decisions

### 1. **Modular Architecture**
- Separation of C++ backend (performance-critical) from Python (flexibility)
- Clear separation of models, losses, data handling, and training logic

### 2. **Production-Ready Features**
- Comprehensive testing (C++ and Python)
- Configuration management (YAML-based)
- Logging and reproducibility
- Documentation (Sphinx)

### 3. **Research Flexibility**
- Multiple model variants (VAE vs Diffusion)
- Configurable loss components
- Easy ablation studies via configs

### 4. **Integration Points**
- R interface for OmniOmicsR integration
- Benchmark suite for comparison with baselines
- Export utilities for downstream analysis

### 5. **Scalability**
- C++ backend for large graphs
- GPU acceleration in PyTorch
- Efficient data loading and batching
- Multi-threaded graph operations

## Build Order

1. **C++ Backend** (cpp_backend/)
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j8
   ```

2. **Python Package** (omnigraphdiff/)
   ```bash
   pip install -e .
   ```

3. **R Interface** (r_interface/)
   ```R
   devtools::install()
   ```

4. **Run Tests**
   ```bash
   pytest tests/
   Rscript -e "devtools::test('r_interface')"
   ```

## Dependencies

### C++ (CMakeLists.txt will handle these)
- Eigen3 (linear algebra)
- pybind11 (Python bindings)
- OpenMP (parallelization)
- Google Test (testing)

### Python (requirements.txt)
- torch >= 2.0.0
- torch-geometric
- numpy, scipy
- pandas
- scikit-learn
- lifelines (survival analysis)
- scanpy (optional, for spatial)
- umap-learn
- seaborn, matplotlib

### R (DESCRIPTION)
- reticulate
- ggplot2
- survival
- OmniOmicsR (integration)
