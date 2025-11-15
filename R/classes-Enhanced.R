# Note: Using OmicsExperiment as base class (already defined in classes-OmicsExperiment.R)

#' SpatialOmicsExperiment class
#'
#' For spatial transcriptomics and spatial proteomics
#' @slot spatial_coords matrix of spatial coordinates (n_spots x 2/3)
#' @slot images list of image data
#' @slot spatial_graphs list of spatial graphs (adjacency/distance)
#' @slot spot_diameter numeric, spot diameter for spatial resolution
#' @slot tissue_positions DataFrame with tissue position info
#' @exportClass SpatialOmicsExperiment
setClass("SpatialOmicsExperiment",
  contains = "OmicsExperiment",
  slots = c(
    spatial_coords = "matrix",
    images = "list",
    spatial_graphs = "list",
    spot_diameter = "numeric",
    tissue_positions = "DataFrame"
  )
)

#' SingleCellMultiOmicsExperiment class
#'
#' For CITE-seq, scATAC-seq, multiome data
#' @slot modalities list of modality names (RNA, ATAC, Protein, etc.)
#' @slot modality_weights numeric vector of modality weights for integration
#' @slot cell_embeddings list of dimensional reductions per modality
#' @slot integrated_embedding matrix of integrated low-D representation
#' @slot trajectory list containing trajectory inference results
#' @slot cell_cell_comm DataFrame of cell-cell communication predictions
#' @exportClass SingleCellMultiOmicsExperiment
setClass("SingleCellMultiOmicsExperiment",
  contains = "OmicsExperiment",
  slots = c(
    modalities = "list",
    modality_weights = "numeric",
    cell_embeddings = "list",
    integrated_embedding = "matrix",
    trajectory = "list",
    cell_cell_comm = "DataFrame"
  )
)

#' ClinicalOmicsProject class
#'
#' Integrates omics with clinical outcomes
#' @slot clinical_data DataFrame with clinical variables
#' @slot survival_data DataFrame with survival times and events
#' @slot treatment_groups factor of treatment assignments
#' @slot biomarkers DataFrame of discovered biomarkers
#' @slot predictions list of clinical prediction model results
#' @slot stratification factor of patient stratification
#' @exportClass ClinicalOmicsProject
setClass("ClinicalOmicsProject",
  contains = "OmniProject",
  slots = c(
    clinical_data = "DataFrame",
    survival_data = "DataFrame",
    treatment_groups = "factor",
    biomarkers = "DataFrame",
    predictions = "list",
    stratification = "factor"
  )
)

#' Constructor for SpatialOmicsExperiment
#' @param x matrix of expression/abundance (features x spots)
#' @param spatial_coords matrix of x,y(,z) coordinates
#' @param omics_type character
#' @param images optional list of images
#' @param ... additional arguments
#' @return SpatialOmicsExperiment
#' @export
create_spatial_experiment <- function(x, spatial_coords, omics_type = "spatial_rna",
                                      images = list(), ...) {
  stopifnot(is.matrix(x), is.matrix(spatial_coords))
  stopifnot(ncol(x) == nrow(spatial_coords))

  se <- SummarizedExperiment::SummarizedExperiment(
    assays = list(counts = x),
    colData = S4Vectors::DataFrame(
      barcode = colnames(x),
      x = spatial_coords[, 1],
      y = spatial_coords[, 2],
      row.names = colnames(x)
    ),
    rowData = S4Vectors::DataFrame(row.names = rownames(x))
  )

  new("SpatialOmicsExperiment", se,
      omics_type = omics_type,
      processing_log = list(list(step = "init", time = Sys.time())),
      feature_map = S4Vectors::DataFrame(),
      spatial_coords = spatial_coords,
      images = images,
      spatial_graphs = list(),
      spot_diameter = 55,  # 10x Visium default
      tissue_positions = S4Vectors::DataFrame())
}

#' Constructor for SingleCellMultiOmicsExperiment
#' @param assays_list named list of matrices (one per modality)
#' @param modality_weights optional weights for each modality
#' @param ... additional arguments
#' @return SingleCellMultiOmicsExperiment
#' @export
create_sc_multiomics <- function(assays_list, modality_weights = NULL, ...) {
  stopifnot(is.list(assays_list), length(assays_list) > 0)

  # Use first modality as primary
  primary <- assays_list[[1]]
  n_cells <- ncol(primary)

  # Verify all modalities have same number of cells
  cell_counts <- sapply(assays_list, ncol)
  if (!all(cell_counts == n_cells)) {
    stop("All modalities must have the same number of cells")
  }

  if (is.null(modality_weights)) {
    modality_weights <- rep(1 / length(assays_list), length(assays_list))
    names(modality_weights) <- names(assays_list)
  }

  se <- SummarizedExperiment::SummarizedExperiment(
    assays = assays_list,
    colData = S4Vectors::DataFrame(
      cell_barcode = colnames(primary),
      row.names = colnames(primary)
    )
  )

  new("SingleCellMultiOmicsExperiment", se,
      omics_type = "sc_multiomics",
      processing_log = list(list(step = "init", time = Sys.time())),
      feature_map = S4Vectors::DataFrame(),
      modalities = as.list(names(assays_list)),
      modality_weights = modality_weights,
      cell_embeddings = list(),
      integrated_embedding = matrix(nrow = n_cells, ncol = 0),
      trajectory = list(),
      cell_cell_comm = S4Vectors::DataFrame())
}

#' Constructor for ClinicalOmicsProject
#' @param omics_assays list of OmicsExperiment objects
#' @param clinical_data DataFrame with clinical variables
#' @param survival_data optional DataFrame with time and event columns
#' @param ... additional arguments
#' @return ClinicalOmicsProject
#' @export
create_clinical_project <- function(omics_assays, clinical_data,
                                    survival_data = NULL, ...) {
  stopifnot(is.list(omics_assays), length(omics_assays) > 0)

  mae <- MultiAssayExperiment::MultiAssayExperiment(experiments = omics_assays)

  if (is.null(survival_data)) {
    survival_data <- S4Vectors::DataFrame(
      time = numeric(0),
      event = integer(0)
    )
  }

  new("ClinicalOmicsProject", mae,
      design = NULL,
      refdb = NULL,
      report = character(0),
      clinical_data = clinical_data,
      survival_data = survival_data,
      treatment_groups = factor(),
      biomarkers = S4Vectors::DataFrame(),
      predictions = list(),
      stratification = factor())
}

#' Show methods
setMethod("show", "SpatialOmicsExperiment", function(object) {
  cat("SpatialOmicsExperiment<", object@omics_type, ">\n", sep = "")
  cat("  Features: ", nrow(object), "\n", sep = "")
  cat("  Spots: ", ncol(object), "\n", sep = "")
  cat("  Coordinates: ", ncol(object@spatial_coords), "D\n", sep = "")
  cat("  Images: ", length(object@images), "\n", sep = "")
})

setMethod("show", "SingleCellMultiOmicsExperiment", function(object) {
  cat("SingleCellMultiOmicsExperiment\n")
  cat("  Cells: ", ncol(object), "\n", sep = "")
  cat("  Modalities: ", paste(unlist(object@modalities), collapse = ", "), "\n", sep = "")
  if (ncol(object@integrated_embedding) > 0) {
    cat("  Integrated embedding: ", ncol(object@integrated_embedding), " dims\n", sep = "")
  }
})

setMethod("show", "ClinicalOmicsProject", function(object) {
  assays <- names(MultiAssayExperiment::experiments(object))
  cat("ClinicalOmicsProject\n")
  cat("  Omics assays: ", paste(assays, collapse = ", "), "\n", sep = "")
  cat("  Patients: ", nrow(object@clinical_data), "\n", sep = "")
  if (nrow(object@survival_data) > 0) {
    cat("  Survival data available\n")
  }
  if (nrow(object@biomarkers) > 0) {
    cat("  Biomarkers identified: ", nrow(object@biomarkers), "\n", sep = "")
  }
})
