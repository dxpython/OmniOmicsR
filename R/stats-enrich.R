#' Map gene IDs using AnnotationDbi
#' @param ids character vector of identifiers
#' @param from source key type (e.g., 'SYMBOL','ENTREZID','ENSEMBL','UNIPROT')
#' @param to target key type
#' @param org_pkg organism annotation package (e.g., 'org.Hs.eg.db')
#' @return data.frame with from/to columns
#' @export
map_ids <- function(ids, from = "SYMBOL", to = "ENTREZID", org_pkg = "org.Hs.eg.db") {
  if (!requireNamespace("AnnotationDbi", quietly = TRUE)) stop("AnnotationDbi not installed")
  if (!requireNamespace(org_pkg, quietly = TRUE)) stop("Organism package not installed: ", org_pkg)
  OrgDb <- get(org_pkg)
  res <- AnnotationDbi::select(OrgDb, keys = unique(ids), keytype = from, columns = c(from, to))
  res <- unique(res[!is.na(res[[to]]), , drop = FALSE])
  res
}

#' KEGG compound mapping via KEGGREST (minimal)
#' @param ids KEGG Compound IDs or names
#' @export
map_kegg_compound <- function(ids) {
  if (!requireNamespace("KEGGREST", quietly = TRUE)) stop("KEGGREST not installed")
  # Minimal: return IDs as-is; real implementation could use keggFind
  data.frame(input = ids, kegg_compound = ids, stringsAsFactors = FALSE)
}

.orgdb_for <- function(organism = c("human","mouse","hsapiens","mmusculus")) {
  organism <- match.arg(organism, several.ok = TRUE)
  if (any(organism %in% c("human","hsapiens"))) return("org.Hs.eg.db")
  if (any(organism %in% c("mouse","mmusculus"))) return("org.Mm.eg.db")
  stop("Unsupported organism: ", organism)
}

.kegg_code_for <- function(organism) {
  if (organism %in% c("human","hsapiens")) return("hsa")
  if (organism %in% c("mouse","mmusculus")) return("mmu")
  stop("Unsupported organism for KEGG: ", organism)
}

#' Pathway enrichment (GO/KEGG/Reactome)
#' @param genes vector of gene IDs
#' @param method one of 'GO','KEGG','Reactome'
#' @param organism 'human' or 'mouse'
#' @param keyType input gene ID type (for GO/Reactome)
#' @param universe background set (optional)
#' @param p_adjust method for multiple testing (default BH)
#' @return enrichment result object from respective package
#' @export
pathway_enrich <- function(genes, method = c("GO", "KEGG", "Reactome"), organism = "human",
                           keyType = "ENTREZID", universe = NULL, p_adjust = "BH") {
  method <- match.arg(method)
  if (method == "GO") {
    if (!requireNamespace("clusterProfiler", quietly = TRUE)) stop("clusterProfiler not installed")
    org_pkg <- .orgdb_for(organism)
    OrgDb <- get(org_pkg)
    res <- clusterProfiler::enrichGO(gene = genes, OrgDb = OrgDb, keyType = keyType,
                                     pAdjustMethod = p_adjust, universe = universe,
                                     ont = "ALL")
    return(res)
  }
  if (method == "KEGG") {
    if (!requireNamespace("clusterProfiler", quietly = TRUE)) stop("clusterProfiler not installed")
    org_code <- .kegg_code_for(organism)
    res <- clusterProfiler::enrichKEGG(gene = genes, organism = org_code, pAdjustMethod = p_adjust, universe = universe)
    return(res)
  }
  if (method == "Reactome") {
    if (!requireNamespace("ReactomePA", quietly = TRUE)) stop("ReactomePA not installed")
    org_name <- if (organism %in% c("human","hsapiens")) "human" else if (organism %in% c("mouse","mmusculus")) "mouse" else stop("Unsupported organism")
    res <- ReactomePA::enrichPathway(gene = genes, organism = org_name, pAdjustMethod = p_adjust, universe = universe)
    return(res)
  }
}

#' Plot enrichment results (dotplot)
#' @param enrich_obj result from clusterProfiler/ReactomePA
#' @param top show top N terms
#' @export
plot_enrichment <- function(enrich_obj, top = 20) {
  if (requireNamespace("enrichplot", quietly = TRUE)) {
    return(enrichplot::dotplot(enrich_obj, showCategory = top))
  }
  df <- tryCatch(as.data.frame(enrich_obj), error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0) stop("Cannot coerce enrichment object to data.frame and enrichplot not installed")
  df <- df[order(df$p.adjust), , drop = FALSE]
  df <- head(df, top)
  df$Term <- factor(df$Description, levels = rev(unique(df$Description)))
  ggplot2::ggplot(df, ggplot2::aes(x = Term, y = -log10(p.adjust))) +
    ggplot2::geom_col() +
    ggplot2::coord_flip() +
    ggplot2::labs(x = NULL, y = "-log10(FDR)", title = "Enrichment") +
    theme_omni()
}
