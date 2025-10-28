#' Differential expression with DESeq2
#'
#' @param object OmicsExperiment with counts assay
#' @param design formula, e.g. ~ group
#' @param contrast optional contrast vector c(var, levelA, levelB)
#' @param assay assay name or index (counts)
#' @param add_to_rowdata logical, attach results to rowData
#' @return list(result=data.frame, dds=DESeqDataSet)
#' @export
dea_deseq2 <- function(object, design = ~ group, contrast = NULL, assay = 1, add_to_rowdata = FALSE) {
  if (!requireNamespace("DESeq2", quietly = TRUE)) stop("DESeq2 not installed")
  x <- SummarizedExperiment::assay(object, assay)
  cd <- as.data.frame(SummarizedExperiment::colData(object))
  dds <- DESeq2::DESeqDataSetFromMatrix(countData = round(x), colData = cd, design = design)
  dds <- DESeq2::DESeq(dds, quiet = TRUE)
  res <- if (is.null(contrast)) DESeq2::results(dds) else DESeq2::results(dds, contrast = contrast)
  df <- as.data.frame(res)
  df$feature <- rownames(df)
  rownames(df) <- rownames(x)
  if (add_to_rowdata) {
    rd <- SummarizedExperiment::rowData(object)
    common <- intersect(colnames(rd), colnames(df))
    if (length(common)) df[common] <- NULL
    SummarizedExperiment::rowData(object) <- cbind(rd, S4Vectors::DataFrame(df))
  }
  .add_log(object, "dea_deseq2", list(assay = assay, contrast = contrast))
  list(result = df, dds = dds)
}

#' Differential expression with edgeR (GLM)
#' @param object OmicsExperiment (counts)
#' @param design_formula formula, e.g. ~ group
#' @param contrast_vector optional numeric contrast vector or limma::makeContrasts-like string is not supported here
#' @param coef optional coefficient index/name if no contrast provided
#' @param assay assay index/name
#' @export
dea_edgeR <- function(object, design_formula = ~ group, contrast_vector = NULL, coef = 2, assay = 1) {
  if (!requireNamespace("edgeR", quietly = TRUE)) stop("edgeR not installed")
  x <- SummarizedExperiment::assay(object, assay)
  cd <- as.data.frame(SummarizedExperiment::colData(object))
  design <- stats::model.matrix(design_formula, data = cd)
  y <- edgeR::DGEList(counts = x)
  y <- edgeR::calcNormFactors(y)
  y <- edgeR::estimateDisp(y, design)
  fit <- edgeR::glmFit(y, design)
  if (!is.null(contrast_vector)) {
    lrt <- edgeR::glmLRT(fit, contrast = contrast_vector)
  } else {
    lrt <- edgeR::glmLRT(fit, coef = coef)
  }
  tt <- edgeR::topTags(lrt, n = Inf, adjust.method = "BH", sort.by = "PValue")
  df <- as.data.frame(tt)
  df$feature <- rownames(df)
  .add_log(object, "dea_edgeR", list(assay = assay))
  list(result = df, fit = fit, lrt = lrt)
}

#' Differential expression with limma (voom optional)
#' @param object OmicsExperiment
#' @param design_formula formula for design
#' @param contrast_matrix optional contrast matrix (columns = contrasts)
#' @param assay assay index or name
#' @param use_voom logical, use voom if counts-like data
#' @param normalize.method normalization method for voom/limma
#' @return list(result, fit)
#' @export
dea_limma <- function(object, design_formula = ~ group, contrast_matrix = NULL, assay = 1,
                      use_voom = FALSE, normalize.method = "quantile") {
  if (!requireNamespace("limma", quietly = TRUE)) stop("limma not installed")
  x <- SummarizedExperiment::assay(object, assay)
  cd <- as.data.frame(SummarizedExperiment::colData(object))
  design <- stats::model.matrix(design_formula, data = cd)
  if (use_voom) {
    if (!requireNamespace("edgeR", quietly = TRUE)) stop("edgeR required for voom")
    y <- edgeR::DGEList(counts = x)
    y <- edgeR::calcNormFactors(y)
    v <- limma::voom(y, design, normalize.method = normalize.method, plot = FALSE)
    fit <- limma::lmFit(v, design)
  } else {
    fit <- limma::lmFit(x, design)
  }
  if (!is.null(contrast_matrix)) {
    fit <- limma::contrasts.fit(fit, contrast_matrix)
  }
  fit <- limma::eBayes(fit)
  tt <- limma::topTable(fit, number = Inf, adjust.method = "BH")
  tt$feature <- rownames(tt)
  .add_log(object, "dea_limma", list(assay = assay, voom = use_voom))
  list(result = tt, fit = fit)
}

