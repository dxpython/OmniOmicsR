#' Read a generic omics matrix
#'
#' Expects the first column as feature IDs and header row with sample IDs.
#' @param expr_file path to csv/tsv
#' @param feature_file optional feature annotation file (ignored in v1 minimal)
#' @param sample_meta optional sample metadata file (ignored in v1 minimal)
#' @param sep field separator, default auto by extension
#' @param omics_type character label
#' @return OmicsExperiment
#' @export
read_omics_matrix <- function(expr_file, feature_file = NULL, sample_meta = NULL,
                              sep = NULL, omics_type = "rna") {
  if (is.null(sep)) {
    sep <- if (grepl("\\.tsv$|\\.txt$", expr_file, ignore.case = TRUE)) "\t" else ","
  }
  dt <- data.table::fread(expr_file, sep = sep)
  df <- as.data.frame(dt)
  stopifnot(ncol(df) >= 2)
  rownames(df) <- as.character(df[[1]])
  df[[1]] <- NULL
  mat <- as.matrix(df)
  storage.mode(mat) <- "double"
  oe <- as_oe(mat, omics_type = omics_type, se_assay_name = "counts")
  .add_log(oe, "import", list(file = expr_file))
}

#' Read MaxQuant proteinGroups output
#'
#' This function detects LFQ or raw Intensity columns and assembles a protein x sample matrix.
#' Contaminants (Reverse/Only identified by site/Potential contaminant) are removed if present.
#' @param file path to proteinGroups.txt
#' @param prefer one of c("LFQ", "Intensity"). If both present, choose this.
#' @param remove_contaminants logical, remove contaminants rows when flags present
#' @param log_transform logical, apply log2(x+1) to intensities
#' @param protein_id_col first non-empty among provided id columns
#' @return OmicsExperiment with assay 'counts' (intensity-like)
#' @export
read_maxquant <- function(file, prefer = c("LFQ", "Intensity"), remove_contaminants = TRUE,
                          log_transform = TRUE,
                          protein_id_col = c("Protein IDs", "Majority protein IDs", "Protein.IDs")) {
  prefer <- match.arg(prefer)
  dt <- data.table::fread(file)
  df <- as.data.frame(dt)
  # Choose protein id column
  id_col <- protein_id_col[protein_id_col %in% colnames(df)][1]
  if (is.na(id_col)) stop("No protein id column found in MaxQuant table.")
  # Remove contaminants if flagged
  flag_cols <- intersect(c("Reverse", "Only identified by site", "Potential contaminant"), colnames(df))
  if (remove_contaminants && length(flag_cols) > 0) {
    bad <- rep(FALSE, nrow(df))
    for (fc in flag_cols) bad <- bad | df[[fc]] %in% c("+", TRUE)
    df <- df[!bad, , drop = FALSE]
  }
  # Detect intensity columns
  lfq_cols <- grep("^LFQ intensity ", colnames(df), value = TRUE)
  int_cols <- grep("^Intensity ", colnames(df), value = TRUE)
  rep_cols <- grep("^Reporter intensity( corrected)? ", colnames(df), value = TRUE)
  cols <- character(0)
  if (prefer == "LFQ" && length(lfq_cols)) cols <- lfq_cols
  if (!length(cols) && length(int_cols)) cols <- int_cols
  if (!length(cols) && length(rep_cols)) cols <- rep_cols
  if (!length(cols)) stop("No intensity columns detected (LFQ/Intensity/Reporter).")
  mat <- as.matrix(df[, cols, drop = FALSE])
  storage.mode(mat) <- "double"
  rownames(mat) <- make.unique(as.character(df[[id_col]]))
  # Clean sample names
  samp <- sub("^(LFQ intensity|Intensity|Reporter intensity( corrected)?) ", "", cols)
  colnames(mat) <- samp
  if (log_transform) mat <- log2(mat + 1)
  oe <- as_oe(mat, omics_type = "protein", se_assay_name = "counts")
  .add_log(oe, "import_maxquant", list(file = file, prefer = prefer, log = log_transform))
}

#' Read mzTab small molecule table (minimal)
#'
#' Expects a tab-delimited table containing small molecule abundance columns
#' like 'smallmolecule_abundance_assay[1]'. Sample names can be provided to map
#' assay indices to names.
#' @param file path to mzTab (tsv)
#' @param sample_names optional character vector; length must match number of abundance columns
#' @param id_cols candidate ID columns
#' @param log_transform apply log2(x+1)
#' @export
read_mztab <- function(file, sample_names = NULL,
                       id_cols = c("identifier", "sml_id", "smallmolecule_identifier"),
                       log_transform = TRUE) {
  dt <- data.table::fread(file, sep = "\t")
  df <- as.data.frame(dt)
  abn_cols <- grep("abundance.*assay\\[[0-9]+\\]$", colnames(df), value = TRUE)
  if (!length(abn_cols)) {
    # fallback: any numeric-only columns beyond first
    num_ix <- which(vapply(df, is.numeric, logical(1)))
    if (length(num_ix) >= 2) abn_cols <- colnames(df)[num_ix]
  }
  if (!length(abn_cols)) stop("No abundance assay columns detected in mzTab table.")
  mat <- as.matrix(df[, abn_cols, drop = FALSE])
  storage.mode(mat) <- "double"
  # sample names
  if (!is.null(sample_names)) {
    stopifnot(length(sample_names) == ncol(mat))
    colnames(mat) <- sample_names
  } else {
    idx <- sub(".*assay\\[([0-9]+)\\]$", "\\1", abn_cols)
    if (any(grepl("[0-9]", idx))) {
      colnames(mat) <- paste0("assay_", idx)
    } else {
      colnames(mat) <- make.names(abn_cols)
    }
  }
  # id column
  id_col <- id_cols[id_cols %in% colnames(df)][1]
  if (!is.na(id_col)) rownames(mat) <- make.unique(as.character(df[[id_col]]))
  if (is.null(rownames(mat))) rownames(mat) <- make.names(seq_len(nrow(mat)))
  if (log_transform) mat <- log2(mat + 1)
  oe <- as_oe(mat, omics_type = "metabolite", se_assay_name = "counts")
  .add_log(oe, "import_mztab", list(file = file, log = log_transform))
}

#' Read Seurat object from RDS (minimal)
#'
#' Requires Seurat. Reads the default assay matrix (data if available, else counts),
#' constructs an OmicsExperiment with colData from meta.data.
#' @param file path to RDS Seurat object
#' @param assay use this Seurat assay if not default
#' @param slot which slot to extract ("data" or "counts")
#' @export
read_seurat <- function(file, assay = NULL, slot = c("data", "counts")) {
  slot <- match.arg(slot)
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    stop("Seurat is not installed. Please install to use read_seurat().")
  }
  obj <- readRDS(file)
  if (!inherits(obj, "Seurat")) stop("Provided RDS does not contain a Seurat object.")
  if (is.null(assay)) assay <- Seurat::DefaultAssay(obj)
  mat <- Seurat::GetAssayData(obj, assay = assay, slot = slot)
  # Ensure dense matrix for SE; keep rownames/colnames
  mat <- as.matrix(mat)
  cd <- S4Vectors::DataFrame(obj@meta.data)
  oe <- as_oe(mat, omics_type = "rna", se_assay_name = slot, col_data = cd)
  .add_log(oe, "import_seurat", list(file = file, assay = assay, slot = slot))
}
