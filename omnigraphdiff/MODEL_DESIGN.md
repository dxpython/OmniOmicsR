# OmniGraphDiff - Mathematical Model Formulation

**Hierarchical Graph-Driven Generative Multi-Omics Integration**

---

## 1. Problem Formulation

### 1.1 Input Data

Given M omics modalities measured on N samples with P features:

- **Omics matrices**: $\mathbf{X}^{(m)} \in \mathbb{R}^{N \times P_m}$, $m = 1, \ldots, M$
  - $\mathbf{X}^{(m)}_{ij}$: Expression/abundance of feature $j$ in sample $i$ for modality $m$

- **Sample metadata**: $\mathbf{S} \in \mathbb{R}^{N \times D_s}$ (clinical covariates, batch, etc.)

- **Feature annotations**: $\mathbf{F}^{(m)} \in \mathbb{R}^{P_m \times D_f}$ (gene sets, pathways, domains)

- **Optional spatial/cell coordinates**: $\mathbf{C} \in \mathbb{R}^{N \times 2}$ or $\mathbb{R}^{N \times 3}$

### 1.2 Objective

Learn a **hierarchical generative model** $p_\theta(\mathbf{X}^{(1)}, \ldots, \mathbf{X}^{(M)} | \mathbf{G})$ that:

1. Captures **cross-modal dependencies** via latent structure $\mathbf{Z} = [\mathbf{Z}_{\text{shared}}, \mathbf{Z}_{\text{spec}}^{(1)}, \ldots, \mathbf{Z}_{\text{spec}}^{(M)}]$
2. Leverages **graph structure** $\mathbf{G} = (\mathcal{G}_{\text{feat}}, \mathcal{G}_{\text{sample}}, \mathcal{G}_{\text{cell}})$ to encode biological priors
3. Supports **generative tasks**: imputation, denoising, perturbation simulation, feature generation
4. Enables **clinical prediction**: survival, stratification, biomarker discovery

---

## 2. Hierarchical Graph Construction

### 2.1 Three-Layer Graph Architecture

#### Layer 1: Feature-Level Graphs $\mathcal{G}_{\text{feat}}^{(m)} = (\mathcal{V}_f^{(m)}, \mathcal{E}_f^{(m)})$

**Nodes**: Features (genes, proteins, metabolites, peaks)

**Edges**: Constructed from:
- **Prior knowledge**: Pathway databases (KEGG, Reactome), protein-protein interactions (STRING), transcription factor targets (TRRUST)
- **Data-driven**: Correlation/mutual information $> \tau$, co-expression modules (WGCNA)

**Adjacency matrix**:
$$
A_f^{(m)}[i,j] =
\begin{cases}
1 & \text{if } (i,j) \in \mathcal{E}_f^{(m)} \\
0 & \text{otherwise}
\end{cases}
$$

**Normalized Laplacian**:
$$
\mathbf{L}_f^{(m)} = \mathbf{I} - \mathbf{D}_f^{-1/2} \mathbf{A}_f^{(m)} \mathbf{D}_f^{-1/2}
$$
where $\mathbf{D}_f$ is the degree matrix.

#### Layer 2: Sample-Level Graph $\mathcal{G}_{\text{sample}} = (\mathcal{V}_s, \mathcal{E}_s)$

**Nodes**: Samples/patients

**Edges**: $k$-NN graph in multi-omics space or metadata space
$$
\mathcal{E}_s = \{(i, j) : j \in \text{kNN}(i, k) \text{ or } \text{sim}(\mathbf{X}_i, \mathbf{X}_j) > \tau_s\}
$$

**Similarity**: Concatenated modality embeddings or clinical similarity

**Normalized adjacency**: $\tilde{\mathbf{A}}_s = \mathbf{D}_s^{-1/2} \mathbf{A}_s \mathbf{D}_s^{-1/2}$

#### Layer 3: Cell/Spatial Graph $\mathcal{G}_{\text{cell}} = (\mathcal{V}_c, \mathcal{E}_c)$ (Optional)

**For single-cell**: Cells as nodes, kNN in reduced dimension (PCA/UMAP)

**For spatial**: Spots/cells as nodes, edges based on physical distance
$$
\mathcal{E}_c = \{(i, j) : \|\mathbf{C}_i - \mathbf{C}_j\| < r\}
$$
where $r$ is radius threshold (e.g., 100 μm)

**Gaussian kernel adjacency**:
$$
\mathbf{A}_c[i,j] = \exp\left(-\frac{\|\mathbf{C}_i - \mathbf{C}_j\|^2}{2\sigma^2}\right)
$$

### 2.2 Cross-Modal Feature Graph

For modalities with clear mappings (e.g., mRNA → protein):
$$
\mathbf{A}_{\text{cross}}[i, j] =
\begin{cases}
1 & \text{if gene } i \text{ encodes protein } j \\
w_{ij} & \text{if weighted mapping available}
\end{cases}
$$

---

## 3. Generative Model Architectures

### 3.1 Option A: Hierarchical Graph-VAE

#### 3.1.1 Encoder Architecture

**Per-modality feature encoder**:
$$
\mathbf{H}_f^{(m)} = \text{GNN}_{\text{feat}}^{(m)}(\mathbf{X}^{(m)}, \mathbf{A}_f^{(m)})
$$

where $\text{GNN}_{\text{feat}}$ is a 2-3 layer Graph Convolutional Network:
$$
\mathbf{H}_f^{(m),(l+1)} = \sigma\left(\tilde{\mathbf{A}}_f^{(m)} \mathbf{H}_f^{(m),(l)} \mathbf{W}^{(l)}\right)
$$

**Sample-level GNN** (operates on sample graph):
$$
\mathbf{H}_s = \text{GNN}_{\text{sample}}(\mathbf{H}_f^{(1)} \oplus \cdots \oplus \mathbf{H}_f^{(M)}, \tilde{\mathbf{A}}_s)
$$

**Cross-modal attention** (for modality fusion):
$$
\mathbf{Z}_{\text{attn}}^{(m)} = \text{Attention}(\mathbf{Q}^{(m)}, [\mathbf{K}^{(1)}, \ldots, \mathbf{K}^{(M)}], [\mathbf{V}^{(1)}, \ldots, \mathbf{V}^{(M)}])
$$

**Latent variable inference**:
$$
q_\phi(\mathbf{Z} | \mathbf{X}^{(1)}, \ldots, \mathbf{X}^{(M)}, \mathbf{G}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{H}_s), \text{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{H}_s)))
$$

Split into **shared** and **specific** components:
$$
\mathbf{Z} = [\underbrace{\mathbf{Z}_{\text{shared}}}_{\text{dim } d_s}, \underbrace{\mathbf{Z}_{\text{spec}}^{(1)}}_{\text{dim } d_1}, \ldots, \underbrace{\mathbf{Z}_{\text{spec}}^{(M)}}_{\text{dim } d_M}]
$$

#### 3.1.2 Decoder Architecture

**Latent → Sample graph embedding**:
$$
\mathbf{H}_s' = \text{GNN}_{\text{dec}}(\mathbf{Z}, \tilde{\mathbf{A}}_s)
$$

**Per-modality reconstruction**:
$$
\hat{\mathbf{X}}^{(m)} = \text{Decoder}^{(m)}(\mathbf{Z}_{\text{shared}} \oplus \mathbf{Z}_{\text{spec}}^{(m)}, \mathbf{A}_f^{(m)})
$$

Decoder can be:
- **GNN-based**: Graph deconvolution layers
- **MLP**: $\hat{\mathbf{X}}^{(m)} = \sigma(\mathbf{W}_3 \sigma(\mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{Z}^{(m)})))$
- **Hybrid**: GNN feature aggregation + MLP reconstruction

#### 3.1.3 VAE Loss Function

$$
\mathcal{L}_{\text{VAE}} = \underbrace{\mathcal{L}_{\text{recon}}}_{\text{Reconstruction}} + \underbrace{\lambda_1 \mathcal{L}_{\text{KL}}}_{\text{KL divergence}} + \underbrace{\lambda_2 \mathcal{L}_{\text{graph}}}_{\text{Graph smoothness}} + \underbrace{\lambda_3 \mathcal{L}_{\text{clinical}}}_{\text{Clinical prediction}} + \underbrace{\lambda_4 \mathcal{L}_{\text{contrastive}}}_{\text{Cross-modal alignment}}
$$

**Component 1: Reconstruction Loss**
$$
\mathcal{L}_{\text{recon}} = \sum_{m=1}^M \mathcal{L}_m(\mathbf{X}^{(m)}, \hat{\mathbf{X}}^{(m)})
$$

where $\mathcal{L}_m$ depends on data type:
- **RNA-seq (counts)**: Negative binomial log-likelihood
  $$
  \mathcal{L}_{\text{NB}} = -\sum_{i,j} \log \text{NB}(x_{ij} | \mu_{ij}, \theta)
  $$
- **Proteomics (continuous)**: Gaussian MSE
  $$
  \mathcal{L}_{\text{MSE}} = \|\mathbf{X}^{(m)} - \hat{\mathbf{X}}^{(m)}\|_F^2
  $$
- **ATAC/ChIP (binary peaks)**: Binary cross-entropy
  $$
  \mathcal{L}_{\text{BCE}} = -\sum_{i,j} x_{ij} \log \hat{x}_{ij} + (1 - x_{ij}) \log(1 - \hat{x}_{ij})
  $$

**Component 2: KL Divergence**
$$
\mathcal{L}_{\text{KL}} = \text{KL}(q_\phi(\mathbf{Z} | \mathbf{X}, \mathbf{G}) \| p(\mathbf{Z}))
$$

Assuming $p(\mathbf{Z}) = \mathcal{N}(0, \mathbf{I})$:
$$
\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{i=1}^d \left(1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2\right)
$$

**Component 3: Graph Regularization (Laplacian Smoothness)**
$$
\mathcal{L}_{\text{graph}} = \sum_{m=1}^M \text{tr}(\mathbf{Z}^{(m)T} \mathbf{L}_f^{(m)} \mathbf{Z}^{(m)}) + \text{tr}(\mathbf{Z}^T \mathbf{L}_s \mathbf{Z})
$$

Encourages latent representations to be smooth over graph structure.

**Component 4: Clinical Prediction Loss** (if survival/outcome data available)
$$
\mathcal{L}_{\text{clinical}} = \mathcal{L}_{\text{Cox}}(\mathbf{Z}_{\text{shared}}, T, \delta) + \mathcal{L}_{\text{CE}}(\mathbf{Z}_{\text{shared}}, y)
$$

- **Cox partial likelihood** for survival:
  $$
  \mathcal{L}_{\text{Cox}} = -\sum_{i:\delta_i=1} \left(\beta^T \mathbf{Z}_i - \log \sum_{j:T_j \geq T_i} \exp(\beta^T \mathbf{Z}_j)\right)
  $$
- **Cross-entropy** for classification outcomes

**Component 5: Contrastive Loss (Cross-Modal Alignment)**
$$
\mathcal{L}_{\text{contrastive}} = \mathcal{L}_{\text{InfoNCE}}(\mathbf{Z}_{\text{shared}}, \{\mathbf{Z}_{\text{spec}}^{(m)}\}_{m=1}^M)
$$

InfoNCE loss to encourage shared latent to align with modality-specific:
$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{Z}_{\text{shared},i}, \mathbf{Z}_{\text{spec},i}^{(m)}) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(\mathbf{Z}_{\text{shared},i}, \mathbf{Z}_{\text{spec},j}^{(m)}) / \tau)}
$$

---

### 3.2 Option B: Graph-Conditioned Diffusion Model

#### 3.2.1 Forward Diffusion Process

Add Gaussian noise over T steps:
$$
q(\mathbf{X}_t^{(m)} | \mathbf{X}_{t-1}^{(m)}) = \mathcal{N}(\mathbf{X}_t^{(m)}; \sqrt{1 - \beta_t} \mathbf{X}_{t-1}^{(m)}, \beta_t \mathbf{I})
$$

Marginal distribution at step $t$:
$$
q(\mathbf{X}_t^{(m)} | \mathbf{X}_0^{(m)}) = \mathcal{N}(\mathbf{X}_t^{(m)}; \sqrt{\bar{\alpha}_t} \mathbf{X}_0^{(m)}, (1 - \bar{\alpha}_t) \mathbf{I})
$$
where $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$.

#### 3.2.2 Reverse Denoising Process (Graph-Conditioned)

Learn $p_\theta(\mathbf{X}_{t-1}^{(m)} | \mathbf{X}_t^{(m)}, \mathbf{G}, \mathbf{c})$ where:
- $\mathbf{G}$: Graph structure (feature + sample graphs)
- $\mathbf{c}$: Conditioning (other modalities, clinical data)

**Noise prediction network**:
$$
\boldsymbol{\epsilon}_\theta(\mathbf{X}_t^{(m)}, t, \mathbf{G}, \mathbf{c}) = \text{GraphUNet}(\mathbf{X}_t^{(m)}, \mathbf{A}_f^{(m)}, \mathbf{A}_s, \mathbf{c})
$$

**Reverse step**:
$$
\mathbf{X}_{t-1}^{(m)} = \frac{1}{\sqrt{\alpha_t}} \left(\mathbf{X}_t^{(m)} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{X}_t^{(m)}, t, \mathbf{G}, \mathbf{c})\right) + \sigma_t \mathbf{z}
$$

#### 3.2.3 Diffusion Training Objective

Simplified objective (similar to DDPM):
$$
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, \mathbf{X}_0, \boldsymbol{\epsilon}} \left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t} \mathbf{X}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t, \mathbf{G}, \mathbf{c})\|^2\right]
$$

**Add graph regularization**:
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}} + \lambda_{\text{graph}} \mathcal{L}_{\text{graph}} + \lambda_{\text{clinical}} \mathcal{L}_{\text{clinical}}
$$

---

## 4. Training Strategy

### 4.1 Two-Stage Training

**Stage 1: Unsupervised Pre-training**
- Optimize $\mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{KL}} + \lambda_2 \mathcal{L}_{\text{graph}}$
- Learn robust multi-omics representations
- Typical epochs: 50-100

**Stage 2: Fine-tuning with Clinical Objectives**
- Add $\mathcal{L}_{\text{clinical}}$ and $\mathcal{L}_{\text{contrastive}}$
- Freeze encoder layers (optional), fine-tune clinical head
- Typical epochs: 20-50

### 4.2 Optimization

**Optimizer**: AdamW with weight decay $10^{-4}$

**Learning rate schedule**:
- Warmup: Linear increase for 1000 steps
- Cosine decay: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t\pi}{T}))$

**Batch size**: 32-128 (depending on memory)

**Gradient clipping**: Max norm = 1.0

### 4.3 Hyperparameter Selection

Use **Optuna** or **Ray Tune** for Bayesian optimization:
- Latent dimensions: $d_{\text{shared}} \in [16, 64]$, $d_{\text{spec}}^{(m)} \in [8, 32]$
- Loss weights: $\lambda_1 \in [0.1, 1.0]$, $\lambda_2 \in [0.01, 0.5]$, $\lambda_3 \in [0.1, 2.0]$
- GNN layers: 2-4 layers
- Hidden dimensions: 128-512

**Validation metric**: Held-out reconstruction loss + C-index (if survival data)

---

## 5. Inference and Downstream Tasks

### 5.1 Latent Representation Extraction

Given trained model, encode new samples:
$$
\mathbf{Z}_{\text{new}} = \text{Encoder}(\mathbf{X}_{\text{new}}^{(1)}, \ldots, \mathbf{X}_{\text{new}}^{(M)}, \mathbf{G})
$$

Use $\mathbf{Z}_{\text{shared}}$ for:
- **Clustering**: k-means, hierarchical, Leiden
- **Visualization**: UMAP, t-SNE
- **Clinical prediction**: Cox regression, Random Forest, XGBoost

### 5.2 Missing Modality Imputation

If modality $m'$ is missing for sample $i$:
1. Encode available modalities: $\mathbf{Z}_i = \text{Encoder}(\{\mathbf{X}_i^{(m)}\}_{m \neq m'})$
2. Sample specific latent: $\mathbf{Z}_{\text{spec},i}^{(m')} \sim p(\mathbf{Z}_{\text{spec}}^{(m')} | \mathbf{Z}_{\text{shared},i})$
3. Decode: $\hat{\mathbf{X}}_i^{(m')} = \text{Decoder}^{(m')}(\mathbf{Z}_{\text{shared},i} \oplus \mathbf{Z}_{\text{spec},i}^{(m')})$

### 5.3 Feature Perturbation Simulation

To simulate perturbation of gene $j$ in modality $m$:
1. Modify feature graph: Remove/add edges to $\mathcal{G}_{\text{feat}}^{(m)}$
2. Generate perturbed latent: $\mathbf{Z}_{\text{pert}} = f_{\text{pert}}(\mathbf{Z}, j, \Delta)$
3. Decode: $\hat{\mathbf{X}}_{\text{pert}} = \text{Decoder}(\mathbf{Z}_{\text{pert}}, \mathbf{G}_{\text{pert}})$

### 5.4 Conditional Generation

Generate samples conditioned on clinical covariates $\mathbf{c}$ (e.g., cancer subtype):
1. Sample latent: $\mathbf{Z} \sim p(\mathbf{Z} | \mathbf{c})$ (class-conditional VAE/diffusion)
2. Decode: $\hat{\mathbf{X}} = \text{Decoder}(\mathbf{Z}, \mathbf{G})$

---

## 6. Model Evaluation

### 6.1 Reconstruction Quality

**Per-modality metrics**:
- **RNA-seq**: Pearson correlation, Spearman correlation, negative binomial deviance
- **Proteomics**: MSE, MAE, R²
- **ATAC-seq**: AUROC, AUPRC for peak calling

**Cross-modal metrics**:
- **Canonical correlation** between modality pairs in latent space
- **Mutual information** between shared and specific latents

### 6.2 Integration Quality

**Clustering metrics** (vs. known cell types/subtypes):
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Silhouette coefficient

**Batch correction** (if multi-batch data):
- kBET (k-nearest neighbor batch effect test)
- LISI (Local Inverse Simpson's Index)

### 6.3 Clinical Prediction

**Survival analysis**:
- **C-index** (concordance index)
- **Integrated Brier Score (IBS)**
- **Log-rank test** p-value for stratification

**Classification** (e.g., cancer subtype):
- **AUROC**, **AUPRC**
- **Balanced accuracy**

### 6.4 Biological Validation

**Pathway enrichment**:
- Top features/modules → Gene Ontology, KEGG enrichment
- Adjusted p-values (Benjamini-Hochberg)

**Known biomarkers**:
- Recovery of established cancer drivers
- Consistency with literature

---

## 7. Comparison to Baselines

### 7.1 Baseline Methods

**Multi-omics integration**:
- **MOFA2**: Multi-Omics Factor Analysis (Bayesian factor model)
- **DIABLO**: Data Integration Analysis for Biomarker discovery using Latent variable approaches (sparse PLS)
- **RGCCA**: Regularized Generalized Canonical Correlation Analysis
- **MultiVI**: Variational inference for single-cell multi-omics (scVI framework)

**Standard dimensionality reduction**:
- **Concatenated PCA**: Naive concatenation + PCA
- **Separate PCA**: Per-modality PCA + concatenation

### 7.2 Benchmark Metrics

Run 5-fold cross-validation on 3-5 datasets (TCGA, CPTAC, 10X multiome):

| Method | Reconstruction (R²) | Integration (ARI) | C-index (Survival) | Runtime (GPU hr) |
|--------|---------------------|-------------------|-------------------|------------------|
| OmniGraphDiff (VAE) | Target > 0.75 | Target > 0.70 | Target > 0.75 | ~2-4 |
| OmniGraphDiff (Diffusion) | Target > 0.78 | Target > 0.72 | Target > 0.76 | ~4-6 |
| MOFA2 | Baseline | Baseline | Baseline | ~1 (CPU) |
| DIABLO | ... | ... | ... | ~0.5 (CPU) |

**Hypothesis**: OmniGraphDiff should outperform by leveraging:
1. Graph structure (vs. linear assumptions)
2. Deep non-linear representations (vs. factor models)
3. Joint generative modeling (vs. correlation-based)

---

## 8. Scalability Considerations

### 8.1 Large Graphs

**Challenge**: Feature graphs with $P > 10,000$ nodes

**Solutions**:
- **Sparse graph ops**: CSR/CSC storage, SpMM in C++
- **Graph sampling**: Sample subgraphs per mini-batch (GraphSAINT, ClusterGCN)
- **Approximate GNN**: SGC (Simple Graph Convolution) removes non-linearity for efficiency

### 8.2 Large Sample Size

**Challenge**: $N > 100,000$ samples (e.g., UK Biobank scale)

**Solutions**:
- **Mini-batch training**: Sample graph neighborhood per batch
- **Multi-GPU**: Data parallelism (PyTorch DDP)
- **Gradient checkpointing**: Trade computation for memory

### 8.3 Memory Optimization

- **Mixed precision training**: FP16 for forward/backward, FP32 for optimizer
- **CPU offloading**: Keep large graphs in CPU, transfer subgraphs to GPU
- **Flash Attention**: For cross-modal attention layers

---

## 9. Software Architecture

### 9.1 C++ Backend (via pybind11)

**Core operations**:
```cpp
// Sparse matrix-matrix multiplication (SpMM)
torch::Tensor sparse_mm(torch::Tensor A_indices, torch::Tensor A_values,
                         torch::Tensor B_dense);

// Laplacian computation
torch::Tensor compute_normalized_laplacian(torch::Tensor adj);

// Graph sampling
std::tuple<torch::Tensor, torch::Tensor> sample_neighbors(
    torch::Tensor adj_indices, torch::Tensor nodes, int k);
```

**Parallelization**: OpenMP for multi-threaded operations

### 9.2 Python/PyTorch Models

**Module hierarchy**:
```
omnigraphdiff.models
├── OmniGraphVAE
│   ├── ModalityEncoder (per modality)
│   ├── CrossModalAttention
│   ├── GraphEncoder (sample-level GNN)
│   ├── LatentSampler (reparameterization)
│   └── ModalityDecoder (per modality)
└── OmniGraphDiffusion
    ├── GraphUNet (noise predictor)
    ├── TimeEmbedding
    └── ConditionEncoder
```

### 9.3 R Interface

Wrapper functions for OmniOmicsR integration:
```r
omnigraph_fit <- function(omniproject, graph_config, model_type = "vae") {
  # Convert OmniProject to Python-compatible format
  # Call omnigraphdiff.train()
  # Return trained model + latent embeddings
}

omnigraph_predict <- function(omnigraph_model, new_omicsexperiment) {
  # Encode new samples
  # Return latent representation
}
```

---

## 10. Expected Contributions

### 10.1 Methodological Novelty

1. **First hierarchical graph-driven generative model** for multi-omics (to our knowledge)
2. **Principled integration** of:
   - Feature-level biological graphs (pathways, PPI)
   - Sample-level similarity graphs
   - Spatial/cellular graphs (for sc/spatial data)
3. **Flexible generative framework**: Both VAE and diffusion variants
4. **Multi-objective optimization**: Balances reconstruction, regularization, and clinical prediction

### 10.2 Practical Advantages

1. **Missing modality imputation**: Handles incomplete multi-omics datasets
2. **Scalability**: C++ backend for 100K+ features
3. **Interpretability**: Graph structure enforces biological constraints
4. **Clinical utility**: Direct integration with survival/outcome prediction

### 10.3 Target Publication Venues

**Computational Biology**: *Nature Methods*, *Nature Machine Intelligence*, *Genome Biology*, *Bioinformatics*

**Machine Learning**: *NeurIPS*, *ICML*, *ICLR* (as methodological ML work)

**Benchmarking Requirements**:
- 3+ real-world datasets (TCGA, CPTAC, spatial/sc)
- Comparison to 5+ baselines
- Ablation studies for each loss component
- Runtime/memory benchmarks
- Biological validation (pathway enrichment, biomarker recovery)

---

## 11. Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
- ✅ C++ sparse graph operations
- ✅ PyTorch GNN layers (GCN, GAT, GraphSAGE)
- ✅ Data loaders for multi-omics + graphs

### Phase 2: Model Implementation (Weeks 3-4)
- ✅ OmniGraphVAE encoder/decoder
- ✅ Loss functions (all 5 components)
- ✅ Training loop with logging

### Phase 3: Evaluation (Weeks 5-6)
- ✅ Benchmark suite (MOFA2, DIABLO, RGCCA wrappers)
- ✅ Metrics calculation
- ✅ Visualization tools

### Phase 4: R Integration (Week 7)
- ✅ reticulate interface
- ✅ OmniOmicsR compatibility layer
- ✅ Vignettes

### Phase 5: Documentation & Release (Week 8)
- ✅ API documentation (Sphinx)
- ✅ Tutorials (Jupyter notebooks)
- ✅ Manuscript writing

---

## References

**Graph Neural Networks**:
- Kipf & Welling (2017) - GCN
- Veličković et al. (2018) - GAT
- Hamilton et al. (2017) - GraphSAGE

**Variational Autoencoders**:
- Kingma & Welling (2014) - VAE
- Lopez et al. (2018) - scVI (for single-cell)

**Diffusion Models**:
- Ho et al. (2020) - DDPM
- Song et al. (2021) - Score-based generative models

**Multi-Omics Integration**:
- Argelaguet et al. (2020) - MOFA2
- Singh et al. (2019) - DIABLO
- Tenenhaus & Tenenhaus (2011) - RGCCA

**Spatial Omics**:
- Cable et al. (2022) - cell2location
- Palla et al. (2022) - Squidpy

---

**End of Mathematical Formulation**

This document provides the complete theoretical foundation for OmniGraphDiff. Implementation details follow in the codebase.
