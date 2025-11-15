#' Variational Autoencoder for omics integration
#'
#' @description
#' Implements VAE for dimensionality reduction and multi-omics integration
#' using keras/tensorflow backend with R interface
#'
#' @param oe OmicsExperiment or matrix
#' @param latent_dim integer, latent dimension size
#' @param hidden_dims integer vector, hidden layer dimensions
#' @param epochs integer, training epochs
#' @param batch_size integer
#' @param learning_rate numeric
#' @param validation_split numeric, fraction for validation
#' @param use_gpu logical, use GPU if available
#' @return list with model, encoder, decoder, and latent representation
#' @export
train_vae <- function(oe, latent_dim = 32, hidden_dims = c(512, 256, 128),
                      epochs = 100, batch_size = 32, learning_rate = 1e-3,
                      validation_split = 0.2, use_gpu = TRUE) {

  # Check if keras is available
  if (!requireNamespace("keras", quietly = TRUE)) {
    warning("keras not available, using PCA approximation")
    return(.vae_pca_fallback(oe, latent_dim))
  }

  # Extract data matrix
  if (inherits(oe, "SummarizedExperiment")) {
    x <- t(SummarizedExperiment::assay(oe, 1))  # samples x features
  } else {
    x <- t(as.matrix(oe))
  }

  # Normalize
  x <- scale(x)
  x[is.na(x)] <- 0
  input_dim <- ncol(x)

  # Build encoder
  encoder_input <- keras::layer_input(shape = input_dim)
  h <- encoder_input

  for (dim in hidden_dims) {
    h <- h |>
      keras::layer_dense(units = dim, activation = "relu") |>
      keras::layer_batch_normalization() |>
      keras::layer_dropout(rate = 0.1)
  }

  z_mean <- h |> keras::layer_dense(units = latent_dim, name = "z_mean")
  z_log_var <- h |> keras::layer_dense(units = latent_dim, name = "z_log_var")

  # Sampling layer
  sampling <- function(args) {
    z_mean <- args[[1]]
    z_log_var <- args[[2]]
    epsilon <- keras::k_random_normal(
      shape = keras::k_shape(z_mean),
      mean = 0, stddev = 1
    )
    z_mean + keras::k_exp(0.5 * z_log_var) * epsilon
  }

  z <- keras::layer_lambda(sampling, output_shape = latent_dim)
  z <- z(list(z_mean, z_log_var))

  encoder <- keras::keras_model(encoder_input, list(z_mean, z_log_var, z))

  # Build decoder
  decoder_input <- keras::layer_input(shape = latent_dim)
  h_decoded <- decoder_input

  for (dim in rev(hidden_dims)) {
    h_decoded <- h_decoded |>
      keras::layer_dense(units = dim, activation = "relu") |>
      keras::layer_batch_normalization() |>
      keras::layer_dropout(rate = 0.1)
  }

  decoder_output <- h_decoded |>
    keras::layer_dense(units = input_dim, activation = "linear")

  decoder <- keras::keras_model(decoder_input, decoder_output)

  # Full VAE
  vae_output <- decoder(z)
  vae <- keras::keras_model(encoder_input, vae_output)

  # VAE loss
  reconstruction_loss <- keras::loss_mean_squared_error(encoder_input, vae_output)
  reconstruction_loss <- reconstruction_loss * input_dim

  kl_loss <- -0.5 * keras::k_sum(
    1 + z_log_var - keras::k_square(z_mean) - keras::k_exp(z_log_var),
    axis = -1L
  )

  vae_loss <- keras::k_mean(reconstruction_loss + kl_loss)
  vae |> keras::add_loss(vae_loss)

  # Compile
  vae |> keras::compile(
    optimizer = keras::optimizer_adam(learning_rate = learning_rate)
  )

  # Train
  history <- vae |> keras::fit(
    x = x,
    y = NULL,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = validation_split,
    verbose = 0
  )

  # Get latent representation
  latent <- encoder |> keras::predict(x)
  z_mean_vals <- latent[[1]]
  rownames(z_mean_vals) <- rownames(x)
  colnames(z_mean_vals) <- paste0("VAE_", 1:latent_dim)

  list(
    vae = vae,
    encoder = encoder,
    decoder = decoder,
    latent = z_mean_vals,
    history = history,
    params = list(
      latent_dim = latent_dim,
      hidden_dims = hidden_dims,
      input_dim = input_dim
    )
  )
}

#' PCA fallback when keras not available
#' @keywords internal
.vae_pca_fallback <- function(oe, latent_dim) {
  if (inherits(oe, "SummarizedExperiment")) {
    x <- t(SummarizedExperiment::assay(oe, 1))
  } else {
    x <- t(as.matrix(oe))
  }

  x <- scale(x)
  x[is.na(x)] <- 0

  pca <- prcomp(x, rank. = latent_dim)

  list(
    vae = NULL,
    encoder = NULL,
    decoder = NULL,
    latent = pca$x,
    history = NULL,
    params = list(latent_dim = latent_dim, method = "PCA_fallback")
  )
}

#' Multi-omics VAE integration
#'
#' @param omics_list list of OmicsExperiment objects
#' @param latent_dim integer
#' @param shared_dim integer, shared latent dimension
#' @param specific_dims integer vector, modality-specific dimensions
#' @param ... additional arguments to train_vae
#' @return list with integrated latent space
#' @export
integrate_vae_multiomics <- function(omics_list, latent_dim = 32,
                                     shared_dim = 16,
                                     specific_dims = NULL, ...) {

  n_modalities <- length(omics_list)

  if (is.null(specific_dims)) {
    specific_dims <- rep((latent_dim - shared_dim) %/% n_modalities, n_modalities)
  }

  # Train individual VAEs
  individual_vaes <- lapply(seq_along(omics_list), function(i) {
    message("Training VAE for modality ", i, "/", n_modalities)
    train_vae(omics_list[[i]], latent_dim = shared_dim + specific_dims[i], ...)
  })

  # Extract latent representations
  latents <- lapply(individual_vaes, function(vae) vae$latent)

  # Integrate using shared dimensions (first shared_dim columns)
  shared_latents <- lapply(latents, function(z) z[, 1:shared_dim, drop = FALSE])
  integrated <- Reduce("+", shared_latents) / n_modalities

  # Combine with modality-specific
  specific_latents <- lapply(seq_along(latents), function(i) {
    if (specific_dims[i] > 0) {
      latents[[i]][, (shared_dim + 1):(shared_dim + specific_dims[i]), drop = FALSE]
    } else {
      NULL
    }
  })

  # Concatenate
  all_latents <- do.call(cbind, c(list(integrated), specific_latents[!sapply(specific_latents, is.null)]))

  list(
    individual_vaes = individual_vaes,
    integrated_latent = all_latents,
    shared_latent = integrated,
    specific_latents = specific_latents,
    params = list(
      shared_dim = shared_dim,
      specific_dims = specific_dims,
      n_modalities = n_modalities
    )
  )
}

#' Extract VAE features
#' @param vae_result result from train_vae
#' @param n_top integer, number of top features to extract per latent dimension
#' @return DataFrame of important features
#' @export
extract_vae_features <- function(vae_result, n_top = 50) {
  if (is.null(vae_result$decoder)) {
    warning("No decoder available, cannot extract features")
    return(S4Vectors::DataFrame())
  }

  # Get decoder weights (latent -> input)
  decoder <- vae_result$decoder
  weights <- keras::get_weights(decoder)

  # First layer weights: latent_dim x hidden_dim
  # Last layer weights: hidden_dim x input_dim
  # We want to trace back importance

  # Simplified: use absolute column sums as importance
  last_weights <- weights[[length(weights) - 1]]  # Last dense layer
  importance <- colSums(abs(last_weights))

  # Top features per latent dimension
  feature_importance <- data.frame(
    feature_idx = 1:length(importance),
    importance = importance
  )

  feature_importance <- feature_importance[order(-feature_importance$importance), ]
  feature_importance <- head(feature_importance, n_top)

  S4Vectors::DataFrame(feature_importance)
}
